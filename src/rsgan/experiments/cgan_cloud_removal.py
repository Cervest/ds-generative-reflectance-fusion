import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.rsgan import build_model, build_dataset
from .experiment import Experiment
from .utils import stack_optical_and_sar
from ..experiments import EXPERIMENTS


@EXPERIMENTS.register('cgan_cloud_removal')
class cGANCloudRemoval(Experiment):
    """Dummy setup to train and evaluate an autoencoder at cloud removal

    Args:
        generator (nn.Module)
        discriminator (nn.Module)
        dataset (CloudRemovalDataset)
        split (list[float]): dataset split ratios in [0, 1] as [train, val]
            or [train, val, test]
        dataloader_kwargs (dict): parameters of dataloaders
        optimizer_kwargs (dict): parameters of optimizer defined in LightningModule.configure_optimizers
        seed (int): random seed (default: None)
    """
    def __init__(self, generator, discriminator, dataset, split, criterion,
                 dataloader_kwargs, optimizer_kwargs, seed=None):
        super().__init__(model=generator,
                         dataset=dataset,
                         split=split,
                         criterion=criterion,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_kwargs=optimizer_kwargs,
                         seed=seed)
        self.discriminator = discriminator

    def forward(self, x):
        return self.generator(x)

    def train_dataloader(self):
        """Implements LightningModule train loader building method
        """
        loader = DataLoader(dataset=self.train_set,
                            collate_fn=stack_optical_and_sar,
                            **self.dataloader_kwargs)
        return loader

    def val_dataloader(self):
        """Implements LightningModule validation loader building method
        """
        loader = DataLoader(dataset=self.val_set,
                            collate_fn=stack_optical_and_sar,
                            **self.dataloader_kwargs)
        return loader

    def configure_optimizers(self):
        """Implements LightningModule optimizer and learning rate scheduler
        building method
        """
        gen_optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs['generator'])
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), **self.optimizer_kwargs['discriminator'])
        return gen_optimizer, disc_optimizer

    def _step_generator(self, source):
        # Forward pass on source domain data
        estimated_target = self(source)
        output_fake_sample = self.discriminator(estimated_target)

        # Compute generator fooling power
        target_real_sample = torch.ones_like(output_fake_sample)
        gen_loss = self.criterion(output_fake_sample, target_real_sample)
        return gen_loss

        # Setup logs dictionnary
        tensorboard_logs = {'Loss/train_generator': gen_loss}
        output = {'loss': gen_loss,
                  'progress_bar': tensorboard_logs,
                  'log': tensorboard_logs}
        return output

    def _step_discriminator(self, source, target):
        # Forward pass on target domain data
        output_real_sample = self.discriminator(target)

        # Compute discriminative power on real samples
        target_real_sample = torch.ones_like(output_real_sample)
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)

        # Generate fake sample + forward pass, note we detach fake samples to not backprop though generator
        estimated_target = self.model(source)
        output_fake_sample = self.discriminator(estimated_target.detach())

        # Compute discriminative power on fake samples
        target_fake_sample = torch.zeros_like(output_fake_sample)
        loss_fake_sample = self.criterion(output_fake_sample, target_fake_sample)

        disc_loss = 0.5 * (loss_real_sample + loss_fake_sample)
        return disc_loss

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
            loss = self._step_generator(source)
            output = self._format_loss(loss, 'Loss/train_generator')
        if optimizer_idx == 1:
            loss = self._step_discriminator(source, target)
            output = self._format_loss(loss, 'Loss/train_discriminator')
        return output

    def on_epoch_end(self):
        """Implements LightningModule end of epoch operations
        """
        # Store into logger batch of images for visualization
        if not hasattr(self.logger, '_logging_images'):
            val_loader = self.val_dataloader()
            source, target = iter(val_loader).next()
            self.logger._logging_images = (source, target)

        # Compute generated samples out of logging images
        source, target = self.logger._logging_images
        with torch.no_grad():
            output = self(source)

        # Log fake-RGB version for visualization
        self.logger.log_images(output[:, :3], tag='Generated', step=self.current_epoch)
        self.logger.log_images(target[:, :3], tag='Target', step=self.current_epoch)

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
        # Run either generator or discriminator training step
        gen_loss = self._step_generator(source)
        disc_loss = self._step_discriminator(source, target)
        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}


    def validation_epoch_end(self, outputs):
        """LightningModule validation epoch end hook

        Args:
            outputs (list[dict]): list of validation steps outputs

        Returns:
            type: dict
        """
        # Average loss and metrics
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        mse = torch.stack([x['mse'] for x in outputs]).mean()

        # Make tensorboard logs and return
        tensorboard_logs = {'Loss/val': loss, 'MSE/val': mse}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def _compute_loss(self, output, target):
        loss = F.smooth_l1_loss(output, target)
        return loss

    def _compute_metrics(self, output, target):
        mse = F.mse_loss(output, target)
        return mse

    def _format_loss(self, loss, name):
        tensorboard_logs = {name: loss}
        output = {'loss': loss,
                  'progress_bar': tensorboard_logs,
                  'log': tensorboard_logs}
        return output

    @property
    def discriminator(self):
        return self._discriminator

    @discriminator.setter
    def discriminator(self, discriminator):
        self._discriminator = discriminator

    @classmethod
    def build(cls, cfg):
        exp_kwargs = {'autoencoder': build_model(cfg['model']),
                      'dataset': build_dataset(cfg['dataset']),
                      'split': list(cfg['dataset']['split'].values()),
                      'optimizer_kwargs': cfg['optimizer'],
                      'dataloader_kwargs': cfg['dataset']['dataloader'],
                      'seed': cfg['experiment']['seed']}
        return cls(**exp_kwargs)
