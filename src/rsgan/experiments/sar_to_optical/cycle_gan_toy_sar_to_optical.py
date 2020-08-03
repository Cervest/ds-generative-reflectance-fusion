import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.rsgan import build_model, build_dataset
from src.rsgan.experiments import EXPERIMENTS
from src.rsgan.experiments.experiment import ToyImageTranslationExperiment


@EXPERIMENTS.register('cycle_gan_toy_sar_to_optical')
class CycleGANToySARToOptical(ToyImageTranslationExperiment):
    """Setup to train and evaluate cycle-consistent generative adversarial
    networks at sar to optical translation on toy dataset

        Domain A : SAR
        Domain B : Optical

    Args:
        generator_AB (nn.Module): generator from domain A to domain B
        generator_BA (nn.Module): generator from domain B to domain A
        discriminator_A (nn.Module): discriminator for domain A images
        discriminator_B (nn.Module): discriminator for domain B images
        dataset (CloudRemovalDataset)
        split (list[float]): dataset split ratios in [0, 1] as [train, val]
            or [train, val, test]
        dataloader_kwargs (dict): parameters of dataloaders
        optimizer_kwargs (dict): parameters of optimizer defined in LightningModule.configure_optimizers
        lr_scheduler_kwargs (dict): paramters of lr scheduler defined in LightningModule.configure_optimizers
        consistency_weight (float): weight of cycle consistency regularization term
        supervision_weight (float): weight of L2 supervision regularization term
        baseline_classifier (sklearn.BaseEstimator): baseline classifier for evaluation
        seed (int): random seed (default: None)
    """
    def __init__(self, generator_AB, generator_BA, discriminator_A, discriminator_B,
                 dataset, split, dataloader_kwargs, optimizer_kwargs, lr_scheduler_kwargs=None,
                 consistency_weight=None, supervision_weight=None, baseline_classifier=None, seed=None):
        super().__init__(model=generator_AB,
                         dataset=dataset,
                         split=split,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_kwargs=optimizer_kwargs,
                         lr_scheduler_kwargs=lr_scheduler_kwargs,
                         criterion=nn.BCELoss(),
                         baseline_classifier=baseline_classifier,
                         seed=seed)
        self.generator_BA = generator_BA
        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B
        self.consistency_weight = consistency_weight
        self.supervision_weight = supervision_weight

    def forward(self, x):
        return self.generator_AB(x)

    def train_dataloader(self):
        """Implements LightningModule train loader building method
        """
        # Make dataloader of (source, target)
        self.train_set.dataset.use_annotations = False

        # Subsample from dataset to avoid having too many similar views from same time serie
        train_set = self._regular_subsample(dataset=self.train_set,
                                            subsampling_rate=5)

        # Instantiate loader
        train_loader_kwargs = self.dataloader_kwargs.copy()
        train_loader_kwargs.update({'dataset': train_set,
                                    'shuffle': True})
        loader = DataLoader(**train_loader_kwargs)
        return loader

    def val_dataloader(self):
        """Implements LightningModule validation loader building method
        """
        # Make dataloader of (source, target)
        self.val_set.dataset.use_annotations = False

        # Instantiate loader
        val_loader_kwargs = self.dataloader_kwargs.copy()
        val_loader_kwargs.update({'dataset': self.val_set})
        loader = DataLoader(**val_loader_kwargs)
        return loader

    def test_dataloader(self):
        """Implements LightningModule test loader building method
        """
        # Make dataloader of (source, target, annotation)
        self.test_set.dataset.use_annotations = True

        # Instantiate loader with batch size = horizon s.t. whole time series are loaded
        test_loader_kwargs = self.dataloader_kwargs.copy()
        test_loader_kwargs.update({'dataset': self.test_set,
                                   'batch_size': self.test_set.dataset.horizon})
        loader = DataLoader(**test_loader_kwargs)
        return loader

    def configure_optimizers(self):
        """Implements LightningModule optimizer and learning rate scheduler
        building method
        """
        # Joint optimizer for both generators
        gen_params = list(self.generator_AB.parameters()) + list(self.generator_BA.parameters())
        optimizer_generator = torch.optim.Adam(gen_params,
                                               **self.optimizer_kwargs['generators'])

        # Separate optimizers for discriminators
        optimizer_discriminator_A = torch.optim.Adam(self.discriminator_A.parameters(),
                                                     **self.optimizer_kwargs['discriminator_A'])
        optimizer_discriminator_B = torch.optim.Adam(self.discriminator_B.parameters(),
                                                     **self.optimizer_kwargs['discriminator_B'])

        # Define optimizers respective learning rate schedulers
        scheduler_generator = torch.optim.lr_scheduler.ExponentialLR(optimizer_generator,
                                                                     **self.lr_scheduler_kwargs['generators'])
        scheduler_discriminator_A = torch.optim.lr_scheduler.ExponentialLR(optimizer_discriminator_A,
                                                                           **self.lr_scheduler_kwargs['discriminator_A'])
        scheduler_discriminator_B = torch.optim.lr_scheduler.ExponentialLR(optimizer_discriminator_B,
                                                                           **self.lr_scheduler_kwargs['discriminator_B'])

        # Make lightning output dictionnary fashion
        self.optimizers = {0: {'optimizer': optimizer_discriminator_A, 'scheduler': scheduler_discriminator_A},
                           1: {'optimizer': optimizer_discriminator_B, 'scheduler': scheduler_discriminator_B},
                           2: {'optimizer': optimizer_generator, 'scheduler': scheduler_generator}}
        return tuple(self.optimizers.values())

    def _step_discriminator(self, source, target, disc_idx):
        """Runs discriminator forward pass and loss computation
        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor
            disc_idx (int): index of discriminator to use {0: domain A, 1: domain B}
        Returns:
            type: dict
        """
        # Forward pass on target domain data with discriminator A or B
        output_real_sample = self.discriminators[disc_idx](target, source)

        # Compute discriminative power on real samples
        target_real_sample = torch.ones_like(output_real_sample)
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)

        # Generate fake sample in A or B + forward pass, we detach fake samples to not backprop though generator
        estimated_target = self.generators[1 - disc_idx](source)
        output_fake_sample = self.discriminators[disc_idx](estimated_target.detach(), source)

        # Compute discriminative power on fake samples
        target_fake_sample = torch.zeros_like(output_fake_sample)
        loss_fake_sample = self.criterion(output_fake_sample, target_fake_sample)
        disc_loss = loss_real_sample + loss_fake_sample

        # Compute classification training metrics
        fooling_rate, precision, recall = self._compute_classification_metrics(output_real_sample, output_fake_sample)
        return disc_loss, fooling_rate, precision, recall

    def _step_generator(self, source, target, gen_idx):
        """Runs generator forward pass and loss computation
        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor
            gen_idx (int): index of generator to use {0: A->B, 1: B->A}
        Returns:
            type: dict
        """
        # Forward pass on source domain data A or B
        estimated_target = self.generators[gen_idx](source)
        output_fake_sample = self.discriminators[1 - gen_idx](estimated_target, source)

        # Compute generator fooling power
        target_real_sample = torch.ones_like(output_fake_sample)
        gen_loss = self.criterion(output_fake_sample, target_real_sample)

        # Compute cycle consistency loss with generator B->A or A->B
        estimated_source = self.generators[1 - gen_idx](estimated_target)
        cycle_loss = F.smooth_l1_loss(source, estimated_source)

        # Compute supervision loss
        mae = F.smooth_l1_loss(estimated_target, target)
        return gen_loss, cycle_loss, mae

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Implements LightningModule training logic
        Args:
            batch (tuple[torch.Tensor]): source, target pairs batch
            batch_idx (int)
            optimizer_idx (int): {0: optimizer_discriminator_A,
                                  1: optimizer_discriminator_B,
                                  2: optimizer_generator_AB,
                                  3: optimizer_generator_BA}
        Returns:
            type: dict
        """
        # Unfold batch
        source, target = batch

        # Run either generators or discriminators forward pass
        if optimizer_idx == 0:
            # Compute domain A discriminator loss
            disc_loss_A, fooling_rate_A, precision_A, recall_A = self._step_discriminator(target, source, 0)
            logs = {'Loss/train_discriminator_A': disc_loss_A,
                    'Metrics/train_fooling_rate_A': fooling_rate_A,
                    'Metric/train_precision_A': precision_A,
                    'Metrics/train_recall_A': recall_A}
            loss = disc_loss_A

        if optimizer_idx == 1:
            # Compute domain B discriminator loss
            disc_loss_B, fooling_rate_B, precision_B, recall_B = self._step_discriminator(source, target, 1)
            logs = {'Loss/train_discriminator_B': disc_loss_B,
                    'Metrics/train_fooling_rate_B': fooling_rate_B,
                    'Metric/train_precision_B': precision_B,
                    'Metrics/train_recall_B': recall_B}
            loss = disc_loss_B

        if optimizer_idx == 2:
            # Compute generator A->B and B->A loss, cycle consistency and supervision
            gen_loss_AB, cycle_loss_AB, mae_AB = self._step_generator(source, target, 0)
            gen_loss_BA, cycle_loss_BA, mae_BA = self._step_generator(target, source, 1)

            loss_AB = gen_loss_AB + self.consistency_weight * cycle_loss_AB + self.supervision_weight * mae_AB
            loss_BA = gen_loss_BA + self.consistency_weight * cycle_loss_BA + self.supervision_weight * mae_BA

            logs = {'Loss/train_generator_AB': gen_loss_AB,
                    'Loss/cycle_loss_AB': cycle_loss_AB,
                    'Loss/mae_AB': mae_AB,
                    'Loss/train_generator_BA': gen_loss_BA,
                    'Loss/cycle_loss_BA': cycle_loss_BA,
                    'Loss/mae_BA': mae_BA}
            loss = loss_AB + loss_BA

        # Make output dict
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
            output_B = self.generator_AB(source)
            output_A = self.generator_BA(target)

        if self.current_epoch == 0:
            # Log input and groundtruth once only at first epoch
            self.logger.log_images(source[:, :3], tag='Domain A - SAR (fake RGB)', step=self.current_epoch)
            self.logger.log_images(target[:, :3], tag='Domain B - Optical (fake RGB)', step=self.current_epoch)

        # Log generated image at current epoch
        self.logger.log_images(output_A[:, :3], tag='Generated Domain A - SAR (fake RGB)', step=self.current_epoch)
        self.logger.log_images(output_B[:, :3], tag='Generated Domain B - Optical (fake RGB)', step=self.current_epoch)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        """Overrides LightningModule.optimizer_step by updating jointly discriminators
        on the one hand and jointly generators on the other hand
        Non-essential lines of code are added here to explicit training logic
        Args:
            epoch (int): current epoch
            batch_idx (int): index of current batch
            optimizer (torch.optim.Optimizer): pytorch optimizer
            optimizer_idx (int): index of optimizer as ordered in output of
                self.configure_optimizers()
        """
        # Don't optimize on A domain discriminator alone, wait to run forward on B
        if optimizer_idx == 0:
            pass

        # Forwards on both discriminators are completed and we can optimize jointly
        if optimizer_idx == 1:
            super().optimizer_step(epoch, batch_idx, self.optimizers[0]['optimizer'], 0)
            super().optimizer_step(epoch, batch_idx, self.optimizers[1]['optimizer'], 1)

        # Optimize on both generators with same optimizer
        if optimizer_idx == 2:
            super().optimizer_step(epoch, batch_idx, optimizer, 2)

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
            self.logger._logging_images = source[:8], target[:8]

        # Run forward pass on generator A -> B and discriminator on B
        gen_loss, cycle_loss, mae = self._step_generator(source, target, 0)
        disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target, 1)

        # Encapsulate scores in torch tensor
        output = torch.Tensor([gen_loss, cycle_loss, mae, disc_loss, fooling_rate, precision, recall])
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
        gen_loss, cycle_loss, mae, disc_loss, fooling_rate, precision, recall = outputs

        # Make logs dict and return  - track discriminator B max loss for validation
        logs = {'Loss/val_generator_AB': gen_loss.item(),
                'Loss/val_discriminator_B': disc_loss.item(),
                'Loss/val_cycle_loss_AB': cycle_loss.item(),
                'Metric/val_mae': mae.item(),
                'Metric/val_fooling_rate': fooling_rate.item(),
                'Metric/val_precision': precision.item(),
                'Metric/val_recall': recall.item()}

        return {'val_loss': disc_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self):
        raise NotImplementedError

    @property
    def generator_AB(self):
        return self.model

    @property
    def generator_BA(self):
        return self._generator_BA

    @property
    def generators(self):
        return {0: self.generator_AB, 1: self.generator_BA}

    @property
    def discriminator_A(self):
        return self._discriminator_A

    @property
    def discriminator_B(self):
        return self._discriminator_B

    @property
    def discriminators(self):
        return {0: self.discriminator_A, 1: self.discriminator_B}

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def consistency_weight(self):
        return self._consistency_weight

    @property
    def supervision_weight(self):
        return self._supervision_weight

    @generator_BA.setter
    def generator_BA(self, generator_BA):
        self._generator_BA = generator_BA

    @discriminator_A.setter
    def discriminator_A(self, discriminator_A):
        self._discriminator_A = discriminator_A

    @discriminator_B.setter
    def discriminator_B(self, discriminator_B):
        self._discriminator_B = discriminator_B

    @optimizers.setter
    def optimizers(self, optimizers):
        self._optimizers = optimizers

    @consistency_weight.setter
    def consistency_weight(self, consistency_weight):
        self._consistency_weight = consistency_weight

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
        build_kwargs = {'generator_AB': build_model(cfg['model']['generator_AB']),
                        'generator_BA': build_model(cfg['model']['generator_BA']),
                        'discriminator_A': build_model(cfg['model']['discriminator_A']),
                        'discriminator_B': build_model(cfg['model']['discriminator_B']),
                        'dataset': build_dataset(cfg['dataset']),
                        'split': list(cfg['dataset']['split'].values()),
                        'optimizer_kwargs': cfg['optimizer'],
                        'lr_scheduler_kwargs': cfg['lr_scheduler'],
                        'dataloader_kwargs': cfg['dataset']['dataloader'],
                        'seed': cfg['experiment']['seed']}
        if test:
            pass
        else:
            build_kwargs.update({'consistency_weight': cfg['experiment']['consistency_weight'],
                                 'supervision_weight': cfg['experiment']['supervision_weight']})
        return build_kwargs
