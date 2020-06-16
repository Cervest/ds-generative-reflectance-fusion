import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rsgan import build_model, build_dataset
from src.rsgan.experiments import EXPERIMENTS
from src.rsgan.experiments.experiment import ImageTranslationExperiment


@EXPERIMENTS.register('cycle_gan_sar_to_optical')
class CycleGANSARToOptical(ImageTranslationExperiment):
    """Short summary.
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
        consistency_weight (float): weight of cycle consistency regularization term
        supervision_weight (float): weight of L2 supervision regularization term
        baseline_classifier (type): Description of parameter `baseline_classifier`.
        baseline_classifier (sklearn.BaseEstimator): baseline classifier for evaluation
        seed (int): random seed (default: None)
    """
    def __init__(self, generator_AB, generator_BA, discriminator_A, discriminator_B,
                 dataset, split, dataloader_kwargs, optimizer_kwargs, consistency_weight=None,
                 supervision_weight=None, baseline_classifier=None, seed=None):
        super().__init__(model=generator_AB,
                         dataset=dataset,
                         split=split,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_kwargs=optimizer_kwargs,
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

    def configure_optimizers(self):
        """Implements LightningModule optimizer and learning rate scheduler
        building method
        """
        optimizer_generator_AB = torch.optim.Adam(self.generator_AB.parameters(),
                                                  **self.optimizer_kwargs['generator_AB'])
        optimizer_generator_BA = torch.optim.Adam(self.generator_BA.parameters(),
                                                  **self.optimizer_kwargs['generator_BA'])
        optimizer_discriminator_A = torch.optim.Adam(self.discriminator_A.parameters(),
                                                     **self.optimizer_kwargs['discriminator_A'])
        optimizer_discriminator_B = torch.optim.Adam(self.discriminator_B.parameters(),
                                                     **self.optimizer_kwargs['discriminator_B'])
        self.optimizers = {0: optimizer_discriminator_A,
                           1: optimizer_discriminator_B,
                           2: optimizer_generator_AB,
                           3: optimizer_generator_BA}
        return optimizer_discriminator_A, optimizer_discriminator_B, optimizer_generator_AB, optimizer_generator_BA

    def _step_discriminator(self, source, target, disc_idx):
        """Runs discriminator forward pass and loss computation
        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor
            disc_idx (int): index of discriminator to use {0: domain A, 1: domain B}
        Returns:
            type: dict
        """
        # Forward pass on target domain data
        output_real_sample = self.discriminators[disc_idx](target, source)

        # Compute discriminative power on real samples
        target_real_sample = torch.ones_like(output_real_sample)
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)

        # Generate fake sample + forward pass, we detach fake samples to not backprop though generator
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
        # Forward pass on source domain data
        estimated_target = self.generators[gen_idx](source)
        output_fake_sample = self.discriminators[1 - gen_idx](estimated_target, source)

        # Compute generator fooling power
        target_real_sample = torch.ones_like(output_fake_sample)
        gen_loss = self.criterion(output_fake_sample, target_real_sample)

        # Compute cycle consistency loss
        estimated_source = self.generators[1 - gen_idx](estimated_target)
        cycle_loss = F.smooth_l1_loss(source, estimated_source)

        # Compute supervision loss
        mse = F.mse_loss(estimated_target, target)
        return gen_loss, cycle_loss, mse

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
            logs_A = {'Loss/train_discriminator_A': disc_loss_A,
                      'Metrics/train_fooling_rate_A': fooling_rate_A,
                      'Metric/train_precision_A': precision_A,
                      'Metrics/train_recall_A': recall_A}

            # Compute domain B discriminator loss
            disc_loss_B, fooling_rate_B, precision_B, recall_B = self._step_discriminator(source, target, 1)
            logs_B = {'Loss/train_discriminator_B': disc_loss_B,
                      'Metrics/train_fooling_rate_B': fooling_rate_B,
                      'Metric/train_precision_B': precision_B,
                      'Metrics/train_recall_B': recall_B}

            # Make output dict
            logs = {**logs_A, **logs_B}
            output = {'loss': disc_loss_A + disc_loss_B,
                      'progress_bar': logs,
                      'log': logs}

        if optimizer_idx == 2:
            # Compute generator A->B loss, cycle consistency and supervision
            gen_loss_AB, cycle_loss_AB, mse_AB = self._step_generator(source, target, 0)
            logs_AB = {'Loss/train_generator_AB': gen_loss_AB,
                       'Loss/cycle_loss_AB': cycle_loss_AB,
                       'Loss/mse_AB': mse_AB}
            loss_AB = gen_loss_AB + self.consistency_weight * cycle_loss_AB + self.supervision_weight * mse_AB

            # Compute generator B->A loss, cycle consistency and supervision
            gen_loss_BA, cycle_loss_BA, mse_BA = self._step_generator(target, source, 1)
            logs_BA = {'Loss/train_generator_BA': gen_loss_BA,
                       'Loss/cycle_loss_BA': cycle_loss_BA,
                       'Loss/mse_BA': mse_BA}
            loss_BA = gen_loss_BA + self.consistency_weight * cycle_loss_BA + self.supervision_weight * mse_BA

            # Make output dict
            logs = {**logs_AB, **logs_BA}
            output = {'loss': loss_AB + loss_BA,
                      'progress_bar': logs,
                      'log': logs}

        if optimizer_idx in [1, 3]:
            # Skip other cases, they are handled in provious conditions
            output = {'loss': torch.Tensor([0])}

        return output

    def on_epoch_end(self):
        """Implements LightningModule end of epoch operations
        """
        # Compute generated samples out of logging images
        source, target = self.logger._logging_images
        with torch.no_grad():
            output_B = self(source)
            output_A = self.generator_BA(target)

        # Log fake-RGB version for visualization
        if self.current_epoch == 0:
            self.logger.log_images(source[:, :3], tag='Source - Optical (fake RGB)', step=self.current_epoch)
            self.logger.log_images(source[:, -3:], tag='Source - SAR (fake RGB)', step=self.current_epoch)
            self.logger.log_images(target[:, :3], tag='Target - Optical (fake RGB)', step=self.current_epoch)
        self.logger.log_images(output_A[:, :3], tag='Generated Source - Optical (fake RGB)', step=self.current_epoch)
        self.logger.log_images(output_A[:, -3:], tag='Generated Source - SAR (fake RGB)', step=self.current_epoch)
        self.logger.log_images(output_B[:, :3], tag='Generated Target - Optical (fake RGB)', step=self.current_epoch)

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        """Overrides backward method to backpropagate gradients only once
        for generator and discriminators
        """
        if optimizer_idx in [0, 2]:
            super().backward(trainer, loss, optimizer, optimizer_idx)
        else:
            pass

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
        # Forwards on both discriminators are completed and we can optimize jointly on same batch
        if optimizer_idx == 0:
            super().optimizer_step(epoch, batch_idx, self.optimizers[0], 0)
            super().optimizer_step(epoch, batch_idx, self.optimizers[1], 1)

        # Don't optimize on B domain discriminator alone, already done above
        if optimizer_idx == 1:
            pass

        # Forwards on both generators are completed and we can optimize jointly on same batch
        if optimizer_idx == 2:
            super().optimizer_step(epoch, batch_idx, self.optimizers[2], 2)
            super().optimizer_step(epoch, batch_idx, self.optimizers[3], 3)

        # Don't optimize on B -> A generator alone, already done above
        if optimizer_idx == 3:
            pass

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
        gen_loss, cycle_loss, mse = self._step_generator(source, target, 0)
        disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target, 1)
        # Encapsulate scores in torch tensor
        output = torch.Tensor([gen_loss, cycle_loss, mse, disc_loss, fooling_rate, precision, recall])
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
        gen_loss, cycle_loss, mse, disc_loss, fooling_rate, precision, recall = outputs

        # Make logs dict and return
        logs = {'Loss/val_generator_AB': gen_loss.item(),
                'Loss/val_discriminator_B': disc_loss.item(),
                'Loss/val_cycle_loss_AB': cycle_loss.item(),
                'Metric/val_mse': mse.item(),
                'Metric/val_fooling_rate': fooling_rate.item(),
                'Metric/val_precision': precision.item(),
                'Metric/val_recall': recall.item()}
        return {'val_loss': gen_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self):
        pass

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
                        'dataloader_kwargs': cfg['dataset']['dataloader'],
                        'seed': cfg['experiment']['seed']}
        if test:
            pass
        else:
            build_kwargs.update({'consistency_weight': cfg['experiment']['consistency_weight'],
                                 'supervision_weight': cfg['experiment']['supervision_weight']})
        return build_kwargs
