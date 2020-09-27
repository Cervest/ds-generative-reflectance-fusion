import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import reduce
from operator import add

from src.deep_reflectance_fusion import build_model, build_dataset
from src.deep_reflectance_fusion.experiments import EXPERIMENTS
from src.deep_reflectance_fusion.experiments.experiment import ImageTranslationExperiment
from src.deep_reflectance_fusion.experiments.utils import collate, process_tensor_for_vis


@EXPERIMENTS.register('cgan_fusion_modis_landsat')
class cGANFusionMODISLandsat(ImageTranslationExperiment):
    """Setup to train and evaluate conditional GANs at predicting Landsat reflectance
        at time step t_i by fusing MODIS reflectance at t_i and Landsat reflectance
        at time step t_{i-1}

    We probe deep generative models capacity at fusing reflectance by conditionning
    GANs on past date fine information about reflectance spatial structure and
    target date coarse information about reflectance value, e.g.

                             +-----------+
               MODIS_t ----->+           +
                             | Generator |---> Predicted_Landsat_t
          Landsat_{t-1}----->+           +
                             +-----------+

    The dataset used in this experiment is consituted of reflectance time
    series from different sites which makes this experiment possible.

    Adversarial networks loss computation given by :
        LossDisc = E_{x~realdata}[-logD(x)] + E_{z~inputs}[-log(1 - D(G(z)))]
        LossGen = E_{z~inputs}[-logD(z)]

    We approximate:
        E_{x~realdata}[-logD(x)] = Avg(CrossEnt_{x:realbatch}(1, D(x)))
        E_{z~inputs}[-log(1 - D(G(z)))] = Avg(CrossEnt_{x:fakebatch}(0, D(x)))
        E_{z~inputs}[-logD(z)] = Avg(CrossEnt_{x:fakebatch}(1, D(x)))


    Args:
        generator (nn.Module)
        discriminator (nn.Module)
        dataset (MODISLandsatReflectanceFusionDataset)
        split (list[float]): dataset split ratios in [0, 1] as [train, val]
            or [train, val, test]
        supervision_weight (float): weight supervision loss term
        dataloader_kwargs (dict): parameters of dataloaders
        optimizer_kwargs (dict): parameters of optimizer defined in LightningModule.configure_optimizers
        lr_scheduler_kwargs (dict): paramters of lr scheduler defined in LightningModule.configure_optimizers
        seed (int): random seed (default: None)
    """
    def __init__(self, generator, discriminator, dataset, split, dataloader_kwargs,
                 optimizer_kwargs, lr_scheduler_kwargs=None, supervision_weight=None,
                 seed=None):
        super().__init__(model=generator,
                         dataset=dataset,
                         split=split,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_kwargs=optimizer_kwargs,
                         lr_scheduler_kwargs=lr_scheduler_kwargs,
                         criterion=nn.BCELoss(),
                         seed=seed)
        self.supervision_weight = supervision_weight
        self.discriminator = discriminator

    def forward(self, x):
        return self.generator(x)

    def train_dataloader(self):
        """Implements LightningModule train loader building method
        """
        # Concatenate all patches datasets into single dataset
        train_set = reduce(add, iter(self.train_set))

        # Instantiate loader
        train_loader_kwargs = self.dataloader_kwargs.copy()
        train_loader_kwargs.update({'dataset': train_set,
                                    'shuffle': True,
                                    'collate_fn': collate.stack_input_frames})
        loader = DataLoader(**train_loader_kwargs)
        return loader

    def val_dataloader(self):
        """Implements LightningModule train loader building method
        """
        # Concatenate all patches datasets into single dataset
        val_set = reduce(add, iter(self.val_set))

        # Instantiate loader
        val_loader_kwargs = self.dataloader_kwargs.copy()
        val_loader_kwargs.update({'dataset': val_set,
                                  'collate_fn': collate.stack_input_frames})
        loader = DataLoader(**val_loader_kwargs)
        return loader

    def test_dataloader(self):
        """Implements LightningModule test loader building method
        """
        # Concatenate all patches datasets into single dataset
        test_set = reduce(add, iter(self.test_set))

        # Instantiate loader with batch size = horizon s.t. full time series are loaded
        test_loader_kwargs = self.dataloader_kwargs.copy()
        test_loader_kwargs.update({'dataset': test_set,
                                   'collate_fn': collate.stack_input_frames})
        loader = DataLoader(**test_loader_kwargs)
        return loader

    def configure_optimizers(self):
        """Implements LightningModule optimizer and learning rate scheduler
        building method
        """
        # Separate optimizers for generator and discriminator
        gen_optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs['generator'])
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), **self.optimizer_kwargs['discriminator'])

        # Separate learning rate schedulers
        gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer,
                                                                  **self.lr_scheduler_kwargs['generator'])
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(disc_optimizer,
                                                                   **self.lr_scheduler_kwargs['discriminator'])

        # Make lightning output dictionnary fashion
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
        pred_target = self(source)
        output_fake_sample = self.discriminator(pred_target, source)

        # Compute generator fooling power
        target_real_sample = torch.ones_like(output_fake_sample)
        gen_loss = self.criterion(output_fake_sample, target_real_sample)

        # Compute image quality metrics
        psnr, ssim, sam = self._compute_iqa_metrics(pred_target, target, reduction='mean')

        # Compute L1 regularization term
        mae = F.smooth_l1_loss(pred_target, target)
        return gen_loss, mae, psnr, ssim, sam

    def _step_discriminator(self, source, target):
        """Runs discriminator forward pass, loss computation and classification
        metrics computation

        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor

        Returns:
            type: dict
        """
        # Forward pass on target domain data
        output_real_sample = self.discriminator(target, source)

        # Compute discriminative power on real samples - label smoothing on positive samples
        target_real_sample = 0.8 + 0.2 * torch.rand_like(output_real_sample)
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)

        # Generate fake sample + forward pass, we detach fake samples to not backprop though generator
        pred_target = self(source)
        output_fake_sample = self.discriminator(pred_target.detach(), source)

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
            gen_loss, mae, psnr, ssim, sam = self._step_generator(source, target)
            logs = {'Loss/train_generator': gen_loss,
                    'Loss/train_mae': mae,
                    'Metric/train_psnr': psnr,
                    'Metric/train_ssim': ssim,
                    'Metric/train_sam': sam}
            loss = gen_loss + self.supervision_weight * mae

        if optimizer_idx == 1:
            disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target)
            logs = {'Loss/train_discriminator': disc_loss,
                    'Metric/train_fooling_rate': fooling_rate,
                    'Metric/train_precision': precision,
                    'Metric/train_recall': recall}
            loss = disc_loss

        # Make lightning fashion output dictionnary
        output = {'loss': loss,
                  'progress_bar': logs,
                  'log': logs}
        return output

    def on_epoch_end(self):
        """Implements LightningModule end of epoch operations
        """
        # Compute generated samples out of logging images
        source, target = self.logger._logging_images
        with torch.no_grad():
            output = self(source)

        if self.current_epoch == 0:
            # Log input and groundtruth once only at first epoch
            self.logger.log_images(process_tensor_for_vis(source[:, [0, 3, 2]], 1, 99), tag='Source - Landsat (B4-B3-B2)', step=self.current_epoch)
            self.logger.log_images(process_tensor_for_vis(source[:, [4, 7, 6]], 1, 99), tag='Source - MODIS (B1-B4-B3)', step=self.current_epoch)
            self.logger.log_images(process_tensor_for_vis(target[:, [0, 3, 2]], 1, 99), tag='Target - Landsat (B4-B3-B2)', step=self.current_epoch)

        # Log generated image at current epoch
        self.logger.log_images(process_tensor_for_vis(output[:, [0, 3, 2]], 1, 99), tag='Generated - Landsat (B4-B3-B2)', step=self.current_epoch)

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

        # Store into logger images for visualization
        if not hasattr(self.logger, '_logging_images'):
            self.logger._logging_images = source, target

        # Run forward pass on generator and discriminator
        gen_loss, mae, psnr, ssim, sam = self._step_generator(source, target)
        disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target)

        # Encapsulate scores in torch tensor
        output = torch.Tensor([gen_loss, mae, psnr, ssim, sam, disc_loss, fooling_rate, precision, recall])
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
        gen_loss, mae, psnr, ssim, sam, disc_loss, fooling_rate, precision, recall = outputs

        # Make tensorboard logs and return
        logs = {'Loss/val_generator': gen_loss.item(),
                'Loss/val_discriminator': disc_loss.item(),
                'Loss/val_mae': mae.item(),
                'Metric/val_psnr': psnr.item(),
                'Metric/val_ssim': ssim.item(),
                'Metric/val_sam': sam.item(),
                'Metric/val_fooling_rate': fooling_rate.item(),
                'Metric/val_precision': precision.item(),
                'Metric/val_recall': recall.item()}

        # Make lightning fashion output dictionnary - track discriminator max loss for validation
        output = {'val_loss': mae,
                  'log': logs,
                  'progress_bar': logs}
        return output

    def test_step(self, batch, batch_idx):
        """Implements LightningModule testing logic

        Args:
            batch (tuple[torch.Tensor]): source, target pairs batch
            batch_idx (int)

        Returns:
            type: dict
        """
        # Unfold batch
        source, target = batch

        # Run forward pass
        pred_target = self(source)

        # Compute IQA metrics
        psnr, ssim, sam = self._compute_iqa_metrics(pred_target, target)
        mae = F.l1_loss(pred_target, target, reduction='none').mean(dim=(0, 2, 3)).cpu()
        mse = F.mse_loss(pred_target, target, reduction='none').mean(dim=(0, 2, 3)).cpu()

        # Encapsulate into torch tensor
        psnr, ssim, sam = torch.Tensor(psnr), torch.Tensor(ssim), torch.Tensor([sam, sam, sam, sam])
        output = torch.stack([mae, mse, psnr, ssim, sam])
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
        mae, mse, psnr, ssim, sam = outputs

        # Make and dump logs
        output = {'test_mae': mae.tolist(),
                  'test_mse': mse.tolist(),
                  'test_psnr': psnr.tolist(),
                  'test_ssim': ssim.tolist(),
                  'test_sam': sam.tolist()}
        return {'log': output}

    @property
    def generator(self):
        return self.model

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def supervision_weight(self):
        return self._supervision_weight

    @discriminator.setter
    def discriminator(self, discriminator):
        self._discriminator = discriminator

    @supervision_weight.setter
    def supervision_weight(self, supervision_weight):
        self._supervision_weight = supervision_weight

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
        if not test:
            build_kwargs.update({'supervision_weight': cfg['experiment']['supervision_weight']})
        return build_kwargs


@EXPERIMENTS.register('residual_cgan_fusion_modis_landsat')
class ResidualcGANFusionMODISLandsat(cGANFusionMODISLandsat):
    """Overrides cGANFusionMODISLandsat by predicting residual between target
    and source Landsat reflectance instead of target reflectance, e.g.

                           +-----------+
             MODIS_t +---->+           |    +---+
                           | Generator +--->+ Σ +-> Predicted_Landsat_t
        Landsat_{t-1}+---->+           |    +-+-+
                  |        +-----------+      ^
                  |                           |
                  +---------------------------+

    """
    def forward(self, x):
        landsat = x[:, :4]
        residual = self.model(x)
        output = landsat + residual
        return output


@EXPERIMENTS.register('ssim_cgan_fusion_modis_landsat')
class SSIMcGANFusionMODISLandsat(cGANFusionMODISLandsat):
    def __init__(self, generator, discriminator, dataset, split, dataloader_kwargs,
                 optimizer_kwargs, lr_scheduler_kwargs=None, supervision_weight_l1=None,
                 supervision_weight_ssim=None, seed=None):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         dataset=dataset,
                         split=split,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_kwargs=optimizer_kwargs,
                         lr_scheduler_kwargs=lr_scheduler_kwargs,
                         supervision_weight=None,
                         seed=seed)
        self.supervision_weight_l1 = supervision_weight_l1
        self.supervision_weight_ssim = supervision_weight_ssim
        from src.losses import SSIM
        self.ssim_criterion = SSIM()

    def _step_generator(self, source, target):
        """Runs generator forward pass and loss computation

        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor

        Returns:
            type: dict
        """
        # Forward pass on source domain data
        pred_target = self(source)
        output_fake_sample = self.discriminator(pred_target, source)

        # Compute generator fooling power
        target_real_sample = torch.ones_like(output_fake_sample)
        gen_loss = self.criterion(output_fake_sample, target_real_sample)

        # Compute image quality metrics
        psnr, ssim, sam = self._compute_iqa_metrics(pred_target, target, reduction='mean')

        # Compute L1 regularization term
        mae = F.smooth_l1_loss(pred_target, target)
        ssim_loss = 1 - self.ssim_criterion(pred_target, target)
        return gen_loss, mae, ssim_loss, psnr, ssim, sam

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
            gen_loss, mae, ssim_loss, psnr, ssim, sam = self._step_generator(source, target)
            logs = {'Loss/train_generator': gen_loss,
                    'Loss/train_mae': mae,
                    'Loss/train_ssim': ssim_loss,
                    'Metric/train_psnr': psnr,
                    'Metric/train_ssim': ssim,
                    'Metric/train_sam': sam}
            loss = gen_loss + self.supervision_weight_l1 * mae + self.supervision_weight_ssim * ssim_loss

        if optimizer_idx == 1:
            disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target)
            logs = {'Loss/train_discriminator': disc_loss,
                    'Metric/train_fooling_rate': fooling_rate,
                    'Metric/train_precision': precision,
                    'Metric/train_recall': recall}
            loss = disc_loss

        # Make lightning fashion output dictionnary
        output = {'loss': loss,
                  'progress_bar': logs,
                  'log': logs}
        return output

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

        # Store into logger images for visualization
        if not hasattr(self.logger, '_logging_images'):
            self.logger._logging_images = source, target

        # Run forward pass on generator and discriminator
        gen_loss, mae, ssim_loss, psnr, ssim, sam = self._step_generator(source, target)
        disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target)

        # Encapsulate scores in torch tensor
        output = torch.Tensor([gen_loss, mae, ssim_loss, psnr, ssim, sam, disc_loss, fooling_rate, precision, recall])
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
        gen_loss, mae, ssim_loss, psnr, ssim, sam, disc_loss, fooling_rate, precision, recall = outputs

        # Make tensorboard logs and return
        logs = {'Loss/val_generator': gen_loss.item(),
                'Loss/val_discriminator': disc_loss.item(),
                'Loss/val_mae': mae.item(),
                'Loss/val_ssim_loss': ssim_loss.item(),
                'Metric/val_psnr': psnr.item(),
                'Metric/val_ssim': ssim.item(),
                'Metric/val_sam': sam.item(),
                'Metric/val_fooling_rate': fooling_rate.item(),
                'Metric/val_precision': precision.item(),
                'Metric/val_recall': recall.item()}

        # Make lightning fashion output dictionnary - track discriminator max loss for validation
        output = {'val_loss': mae,
                  'log': logs,
                  'progress_bar': logs}
        return output

    @property
    def supervision_weight_l1(self):
        return self._supervision_weight_l1

    @property
    def supervision_weight_ssim(self):
        return self._supervision_weight_ssim

    @supervision_weight_l1.setter
    def supervision_weight_l1(self, supervision_weight_l1):
        self._supervision_weight_l1 = supervision_weight_l1

    @supervision_weight_ssim.setter
    def supervision_weight_ssim(self, supervision_weight_ssim):
        self._supervision_weight_ssim = supervision_weight_ssim

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
        if not test:
            build_kwargs.update({'supervision_weight_l1': cfg['experiment']['supervision_weight_l1'],
                                 'supervision_weight_ssim': cfg['experiment']['supervision_weight_ssim']})
        return build_kwargs
