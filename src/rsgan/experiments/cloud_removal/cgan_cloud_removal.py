import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.rsgan import build_model, build_dataset
from src.rsgan.experiments import EXPERIMENTS
from src.rsgan.experiments.experiment import ImageTranslationExperiment
from src.rsgan.experiments.utils import collate
from src.utils import load_pickle


@EXPERIMENTS.register('cgan_cloud_removal')
class cGANCloudRemoval(ImageTranslationExperiment):
    """Dummy setup to train and evaluate an autoencoder at cloud removal

    Args:
        generator (nn.Module)
        discriminator (nn.Module)
        dataset (CloudRemovalDataset)
        split (list[float]): dataset split ratios in [0, 1] as [train, val]
            or [train, val, test]
        l1_weight (float): weight of l1 regularization term
        dataloader_kwargs (dict): parameters of dataloaders
        optimizer_kwargs (dict): parameters of optimizer defined in LightningModule.configure_optimizers
        lr_scheduler_kwargs (dict): paramters of lr scheduler defined in LightningModule.configure_optimizers
        baseline_classifier (sklearn.BaseEstimator):baseline classifier for evaluation
        seed (int): random seed (default: None)
    """
    def __init__(self, generator, discriminator, dataset, split, dataloader_kwargs,
                 optimizer_kwargs, lr_scheduler_kwargs=None, l1_weight=None,
                 baseline_classifier=None, seed=None):
        super().__init__(model=generator,
                         dataset=dataset,
                         split=split,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_kwargs=optimizer_kwargs,
                         lr_scheduler_kwargs=lr_scheduler_kwargs,
                         criterion=nn.BCELoss(),
                         baseline_classifier=baseline_classifier,
                         seed=seed)
        self.l1_weight = l1_weight
        self.discriminator = discriminator

    def forward(self, x):
        return self.generator(x)

    def train_dataloader(self):
        """Implements LightningModule train loader building method
        """
        # Make dataloader of (source, target)
        self.train_set.dataset.use_annotations = False
        loader = DataLoader(dataset=self.train_set,
                            collate_fn=collate.stack_optical_and_sar,
                            **self.dataloader_kwargs)
        return loader

    def val_dataloader(self):
        """Implements LightningModule validation loader building method
        """
        # Make dataloader of (source, target)
        self.val_set.dataset.use_annotations = False
        loader = DataLoader(dataset=self.val_set,
                            collate_fn=collate.stack_optical_and_sar,
                            **self.dataloader_kwargs)
        return loader

    def test_dataloader(self):
        """Implements LightningModule test loader building method
        """
        # Make dataloader of (source, target, annotation)
        self.test_set.dataset.use_annotations = True
        loader = DataLoader(dataset=self.test_set,
                            collate_fn=collate.stack_optical_sar_and_annotations,
                            **self.dataloader_kwargs)
        return loader

    def configure_optimizers(self):
        """Implements LightningModule optimizer and learning rate scheduler
        building method
        """
        gen_optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs['generator'])
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), **self.optimizer_kwargs['discriminator'])
        gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer,
                                                                  **self.lr_scheduler_kwargs['generator'])
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(disc_optimizer,
                                                                   **self.lr_scheduler_kwargs['discriminator'])
        gen_optimizer_dict = {'optimizer': gen_optimizer, 'scheduler': gen_lr_scheduler, 'frequency': 1}
        disc_optimizer_dict = {'optimizer': disc_optimizer, 'scheduler': disc_lr_scheduler, 'frequency': 2}
        return gen_optimizer_dict, disc_optimizer_dict

    def _step_generator(self, source, target):
        """Runs generator forward pass and loss computation

        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor

        Returns:
            type: dict
        """
        # Forward pass on source domain data
        estimated_target = self(source)
        output_fake_sample = self.discriminator(estimated_target, source)

        # Compute generator fooling power
        target_real_sample = torch.ones_like(output_fake_sample)
        gen_loss = self.criterion(output_fake_sample, target_real_sample)

        # Compute L1 regularization term
        mae = F.smooth_l1_loss(estimated_target, target)
        return gen_loss, mae

    def _step_discriminator(self, source, target):
        """Runs discriminator forward pass and loss computation

        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor

        Returns:
            type: dict
        """
        # Forward pass on target domain data
        output_real_sample = self.discriminator(target, source)

        # Compute discriminative power on real samples
        target_real_sample = torch.ones_like(output_real_sample)
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)

        # Generate fake sample + forward pass, we detach fake samples to not backprop though generator
        estimated_target = self.model(source)
        output_fake_sample = self.discriminator(estimated_target.detach(), source)

        # Compute discriminative power on fake samples
        target_fake_sample = torch.zeros_like(output_fake_sample)
        loss_fake_sample = self.criterion(output_fake_sample, target_fake_sample)
        disc_loss = loss_real_sample + loss_fake_sample

        # Compute classification training metrics
        fooling_rate, precision, recall = self._compute_classification_metrics(output_real_sample, output_fake_sample)
        return disc_loss, fooling_rate, precision, recall

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Implements LightningModule training logic

        Args:
            batch (tuple[torch.Tensor]): source, target pairs batch
            batch_idx (int)
            optimizer_idx (int): {0: gen_optimizer, 1: disc_optimizer}

        Returns:
            type: dict
        """
        # Unfold batch
        source, target = batch
        # Run either generator or discriminator training step
        if optimizer_idx == 0:
            gen_loss, mae = self._step_generator(source, target)
            # Setup logs dictionnary
            tensorboard_logs = {'Loss/train_generator': gen_loss,
                                'Metric/train_mae': mae}
            output = {'loss': gen_loss + self.l1_weight * mae,
                      'progress_bar': tensorboard_logs,
                      'log': tensorboard_logs}
        if optimizer_idx == 1:
            disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target)
            # Setup logs dictionnary
            tensorboard_logs = {'Loss/train_discriminator': disc_loss,
                                'Metric/train_fooling_rate': fooling_rate,
                                'Metric/train_precision': precision,
                                'Metric/train_recall': recall}
            output = {'loss': disc_loss,
                      'progress_bar': tensorboard_logs,
                      'log': tensorboard_logs}
        return output

    def on_epoch_end(self):
        """Implements LightningModule end of epoch operations
        """
        # Compute generated samples out of logging images
        source, target = self.logger._logging_images
        with torch.no_grad():
            output = self(source)

        # Log fake-RGB version for visualization
        if self.current_epoch == 0:
            self.logger.log_images(source[:, :3], tag='Source - Optical (fake RGB)', step=self.current_epoch)
            self.logger.log_images(source[:, -3:], tag='Source - SAR (fake RGB)', step=self.current_epoch)
            self.logger.log_images(target[:, :3], tag='Target - Optical (fake RGB)', step=self.current_epoch)
        self.logger.log_images(output[:, :3], tag='Generated - Optical (fake RGB)', step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        """Implements LightningModule validation logic

        Args:
            batch (tuple[torch.Tensor]): source, target pairs batch
            batch_idx (int)

        Returns:
            type: dict
        """
        # Unfold batch
        source, target = batch
        # Store into logger batch if images for visualization
        if not hasattr(self.logger, '_logging_images'):
            self.logger._logging_images = source[:8], target[:8]
        # Run forward pass on generator and discriminator
        gen_loss, mae = self._step_generator(source, target)
        disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target)
        # Encapsulate scores in torch tensor
        output = torch.Tensor([gen_loss, mae, disc_loss, fooling_rate, precision, recall])
        return output

    def validation_epoch_end(self, outputs):
        """LightningModule validation epoch end hook

        Args:
            outputs (list[torch.Tensor]): list of validation steps outputs

        Returns:
            type: dict
        """
        # Average loss and metrics
        outputs = torch.stack(outputs).mean(dim=0)
        gen_loss, mae, disc_loss, fooling_rate, precision, recall = outputs

        # Make tensorboard logs and return
        tensorboard_logs = {'Loss/val_generator': gen_loss.item(),
                            'Loss/val_discriminator': disc_loss.item(),
                            'Metric/val_mae': mae.item(),
                            'Metric/val_fooling_rate': fooling_rate.item(),
                            'Metric/val_precision': precision.item(),
                            'Metric/val_recall': recall.item()}
        return {'val_loss': gen_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """Implements LightningModule testing logic

        Args:
            batch (tuple[torch.Tensor]): source, target pairs batch
            batch_idx (int)

        Returns:
            type: dict
        """
        # Unfold batch
        source, target, annotation = batch

        # Run generator forward pass
        generated_target = self(source)

        # Compute performance at downstream classification task
        iou_generated, iou_real = self._compute_legitimacy_at_task_score(self.baseline_classifier,
                                                                         generated_target,
                                                                         target,
                                                                         annotation)
        iou_ratio = iou_generated / iou_real

        # Compute IQA metrics
        psnr, ssim, cw_ssim = self._compute_iqa_metrics(generated_target, target)
        mse = F.mse_loss(generated_target, target)
        mae = F.l1_loss(generated_target, target)

        # Encapsulate into torch tensor
        output = torch.Tensor([mae, mse, psnr, ssim, cw_ssim, iou_generated, iou_real, iou_ratio])
        return output

    def test_epoch_end(self, outputs):
        """LightningModule test epoch end hook

        Args:
            outputs (list[torch.Tensor]): list of test steps outputs

        Returns:
            type: dict
        """
        # Average metrics
        outputs = torch.stack(outputs).mean(dim=0)
        mae, mse, psnr, ssim, cw_ssim, iou_estimated, iou_real, iou_ratio = outputs

        # Make and dump logs
        output = {'test_mae': mae.item(),
                  'test_mse': mse.item(),
                  'test_psnr': psnr.item(),
                  'test_ssim': ssim.item(),
                  'test_cw_ssim': cw_ssim.item(),
                  'test_jaccard_generated_samples': iou_estimated.item(),
                  'test_jaccard_real_samples': iou_real.item(),
                  'test_jaccard_ratio': iou_ratio.item()}
        return {'log': output}

    @property
    def generator(self):
        return self.model

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def l1_weight(self):
        return self._l1_weight

    @discriminator.setter
    def discriminator(self, discriminator):
        self._discriminator = discriminator

    @l1_weight.setter
    def l1_weight(self, l1_weight):
        self._l1_weight = l1_weight

    @classmethod
    def _make_build_kwargs(self, cfg, test=False):
        """Build keyed arguments dictionnary out of configurations to be passed
            to class constructor

        Args:
            cfg (dict): loaded YAML configuration file
            test (bool): set to True for testing

        Returns:
            type: dict
        """
        build_kwargs = {'generator': build_model(cfg['model']['generator']),
                        'discriminator': build_model(cfg['model']['discriminator']),
                        'dataset': build_dataset(cfg['dataset']),
                        'split': list(cfg['dataset']['split'].values()),
                        'optimizer_kwargs': cfg['optimizer'],
                        'lr_scheduler_kwargs': cfg['lr_scheduler'],
                        'dataloader_kwargs': cfg['dataset']['dataloader'],
                        'seed': cfg['experiment']['seed']}
        if test:
            baseline_classifier = load_pickle(cfg['testing']['baseline_classifier_path'])
            build_kwargs.update({'baseline_classifier': baseline_classifier})
        else:
            build_kwargs.update({'l1_weight': cfg['experiment']['l1_regularization_weight']})
        return build_kwargs
