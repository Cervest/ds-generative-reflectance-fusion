import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import reduce
from operator import add

from src.rsgan import build_model, build_dataset
from src.rsgan.experiments import EXPERIMENTS
from src.rsgan.experiments.experiment import ImageTranslationExperiment
from src.rsgan.experiments.utils import collate
from .utils import process_tensor_for_vis


@EXPERIMENTS.register('early_fusion_modis_landsat')
class EarlyFusionMODISLandsat(ImageTranslationExperiment):
    """TODO : Add description

    Args:
        model (nn.Module)
        dataset (MODISLandsatTemporalResolutionFusionDataset)
        split (list[float]): dataset split ratios in [0, 1] as [train, val]
            or [train, val, test]
        dataloader_kwargs (dict): parameters of dataloaders
        optimizer_kwargs (dict): parameters of optimizer defined in LightningModule.configure_optimizers
        lr_scheduler_kwargs (dict): paramters of lr scheduler defined in LightningModule.configure_optimizers
        seed (int): random seed (default: None)
    """
    def __init__(self, model, dataset, split, dataloader_kwargs,
                 optimizer_kwargs, lr_scheduler_kwargs=None, seed=None):
        super().__init__(model=model,
                         dataset=dataset,
                         split=split,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_kwargs=optimizer_kwargs,
                         lr_scheduler_kwargs=lr_scheduler_kwargs,
                         criterion=nn.SmoothL1Loss(),
                         seed=seed)

    def forward(self, x):
        return self.model(x)

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
        # Setup unet optimizer
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)

        # Separate learning rate schedulers
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                              **self.lr_scheduler_kwargs)
        # Make lightning output dictionnary fashion
        optimizer_dict = {'optimizer': optimizer, 'scheduler': lr_scheduler}
        return optimizer_dict

    def training_step(self, batch, batch_idx):
        """Implements LightningModule training logic

        Args:
            batch (tuple[torch.Tensor]): source, target pairs batch
            batch_idx (int)

        Returns:
            type: dict
        """
        # Unfold batch
        source, target = batch

        # Run forward pass + compute MAE loss
        pred_target = self(source)
        loss = self.criterion(pred_target, target)

        # Compute image quality metrics
        psnr, ssim, sam = self._compute_iqa_metrics(pred_target, target)

        # Make lightning fashion output dictionnary
        logs = {'Loss/train_mae': loss,
                'Metric/train_psnr': psnr,
                'Metric/train_ssim': ssim,
                'Metric/train_sam': sam}

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
            self.logger.log_images(process_tensor_for_vis(source[:, [0, 2, 3]], 33, 98), tag='Source - Landsat (B4-B2-B3)', step=self.current_epoch)
            self.logger.log_images(process_tensor_for_vis(source[:, [4, 6, 7]], 0, 98), tag='Source - MODIS (B1-B3-B4)', step=self.current_epoch)
            self.logger.log_images(process_tensor_for_vis(target[:, [0, 2, 3]], 33, 98), tag='Target - Landsat (B4-B2-B3)', step=self.current_epoch)

        # Log generated image at current epoch
        self.logger.log_images(process_tensor_for_vis(output[:, [0, 2, 3]], 33, 98), tag='Generated - Landsat (B4-B2-B3)', step=self.current_epoch)

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

        # Run forward pass
        pred_target = self(source)
        loss = self.criterion(pred_target, target)
        psnr, ssim, sam = self._compute_iqa_metrics(pred_target, target)

        # Encapsulate scores in torch tensor
        output = torch.Tensor([loss, psnr, ssim, sam])
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
        loss, psnr, ssim, sam = outputs

        # Make lightning fashion output dictionnary
        logs = {'Loss/val_mae': loss.item(),
                'Metric/val_psnr': psnr.item(),
                'Metric/val_ssim': ssim.item(),
                'Metric/val_sam': sam.item()}

        output = {'val_loss': loss,
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
        mae = F.l1_loss(pred_target, target)
        mse = F.mse_loss(pred_target, target)

        # Encapsulate into torch tensor
        output = torch.Tensor([mae, mse, psnr, ssim, sam])
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
        output = {'test_mae': mae.item(),
                  'test_mse': mse.item(),
                  'test_psnr': psnr.item(),
                  'test_ssim': ssim.item(),
                  'test_sam': sam.item()}
        return {'log': output}

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
        build_kwargs = {'model': build_model(cfg['model']),
                        'dataset': build_dataset(cfg['dataset']),
                        'split': list(cfg['dataset']['split'].values()),
                        'optimizer_kwargs': cfg['optimizer'],
                        'lr_scheduler_kwargs': cfg['lr_scheduler'],
                        'dataloader_kwargs': cfg['dataset']['dataloader'],
                        'seed': cfg['experiment']['seed']}
        return build_kwargs


@EXPERIMENTS.register('residual_early_fusion_modis_landsat')
class ResidualEarlyFusionMODISLandsat(EarlyFusionMODISLandsat):
    def forward(self, x):
        landsat = x[:, :4]
        residual = self.model(x)
        output = landsat + residual
        return output
