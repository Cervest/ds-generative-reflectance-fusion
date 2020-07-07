import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.rsgan.experiments import EXPERIMENTS
from src.rsgan.experiments.utils import collate
from .cgan_toy_cloud_removal import cGANToyCloudRemoval


@EXPERIMENTS.register('cgan_frame_recurrent_toy_cloud_removal')
class cGANFrameRecurrentToyCloudRemoval(cGANToyCloudRemoval):
    """Setup to train and evaluate conditional GANs at cloud removal on toy dataset
        using past frame prediction to enforce temporal consistency

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
    @staticmethod
    def _reorder_dataset_indices(dataset, batch_size, horizon, temporal_resolution):
        """Reorders dataset indices such that batches of time serie frames
        from same time step t are loaded at the same time when training

        Args:
            dataset (Dataset)
            batch_size (int)
            horizon (int): length of time series
            temporal_resolution (int): subsampling rate of time serie temporal resolution

        Returns:
            type: Dataset, int
        """
        # Initialize array of indices of size (n_time_series, horizon)
        indices = np.arange(len(dataset)).reshape(-1, horizon)

        # Remove intermediary time steps to lower temporal resolution and compute new horizon
        indices = indices[:, ::temporal_resolution]
        new_horizon = np.ceil(horizon / temporal_resolution).astype(int)

        # Truncate number of time series s.t. n_times_series multiple of batch_size
        n_time_series = indices.shape[0]
        n_time_series = n_time_series - n_time_series % batch_size
        indices = indices[:n_time_series]

        # Stack consecutive batches to rearrange array as (batch_size, n_batch * horizon)
        n_batch = n_time_series / batch_size
        indices = np.hstack(np.split(indices, n_batch))

        # Flatten array to feed to subset instance
        indices = indices.transpose().flatten().tolist()
        reordered_dataset = Subset(dataset=dataset, indices=indices)
        return reordered_dataset, new_horizon

    def train_dataloader(self):
        """Implements LightningModule train loader building method
        """
        # Make dataloader of (source, target) - no annotation needed
        self.train_set.dataset.use_annotations = False

        # Reorder indices s.t. batches contain successive frames of same time series
        dataset, new_horizon = self._reorder_dataset_indices(dataset=self.train_set,
                                                             batch_size=self.dataloader_kwargs['batch_size'],
                                                             horizon=self.train_set.dataset.horizon,
                                                             temporal_resolution=5)
        self.horizon = new_horizon

        # Instantiate loader - do not shuffle to respect datatset reordering
        train_loader_kwargs = self.dataloader_kwargs.copy()
        train_loader_kwargs.update({'dataset': dataset,
                                    'collate_fn': collate.stack_optical_with_sar})
        loader = DataLoader(**train_loader_kwargs)
        return loader

    def val_dataloader(self):
        """Implements LightningModule validation loader building method
        """
        # Make dataloader of (source, target) - no annotation needed
        self.val_set.dataset.use_annotations = False

        # Reorder indices s.t. batches contain successive frames of same time series
        dataset, new_horizon = self._reorder_dataset_indices(dataset=self.val_set,
                                                             batch_size=self.dataloader_kwargs['batch_size'],
                                                             horizon=self.val_set.dataset.horizon,
                                                             temporal_resolution=5)
        self.horizon = new_horizon

        # Instantiate loader
        val_loader_kwargs = self.dataloader_kwargs.copy()
        val_loader_kwargs.update({'dataset': dataset,
                                  'collate_fn': collate.stack_optical_with_sar})
        loader = DataLoader(**val_loader_kwargs)
        return loader

    def test_dataloader(self):
        """Implements LightningModule test loader building method
        """
        # Make dataloader of (source, target, annotation)
        self.test_set.dataset.use_annotations = True

        # Instantiate loader with batch size = horizon s.t. full time series are loaded
        test_loader_kwargs = self.dataloader_kwargs.copy()
        test_loader_kwargs.update({'dataset': self.test_set,
                                   'batch_size': self.test_set.dataset.horizon,
                                   'collate_fn': collate.stack_optical_sar_and_annotations})
        loader = DataLoader(**test_loader_kwargs)
        return loader

    def _store_previous_batch_output(self, estimated_target, batch_idx):
        """Stores output of generator on previous batch, if batch is last
        from time serie, stores zero tensor instead

        Args:
            estimated_target (torch.Tensor): (B, C, H, W) tensor output from
                generator on previous batch
            batch_idx (int): index of batch
        """
        # Store zero tensor if this was last batch from time serie
        if batch_idx % self.horizon == self.horizon - 1:
            self._previous_batch_output = torch.zeros_like(estimated_target)
        # Else store generator prediction on last batch
        else:
            self._previous_batch_output = estimated_target.detach()

    def _step_generator(self, source, target):
        """Runs generator forward pass and loss computation

        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor

        Returns:
            type: dict
        """
        # Forward pass on source domain data
        source_and_previous_prediction = torch.cat([source, self._previous_batch_output], dim=1)
        estimated_target = self(source_and_previous_prediction)
        output_fake_sample = self.discriminator(estimated_target, source)

        # Compute generator fooling power
        target_real_sample = torch.ones_like(output_fake_sample)
        gen_loss = self.criterion(output_fake_sample, target_real_sample)

        # Compute L1 regularization term
        mae = F.smooth_l1_loss(estimated_target, target)
        return gen_loss, mae

    def _step_discriminator(self, source, target, batch_idx):
        """Runs discriminator forward pass and loss computation

        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor
            batch_idx (int)

        Returns:
            type: dict
        """
        # Forward pass on target domain data
        output_real_sample = self.discriminator(target, source)

        # Compute discriminative power on real samples
        target_real_sample = torch.ones_like(output_real_sample)
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)

        # Generate fake sample + forward pass, we detach fake samples to not backprop though generator
        source_and_previous_prediction = torch.cat([source, self._previous_batch_output], dim=1)
        estimated_target = self.model(source_and_previous_prediction)
        output_fake_sample = self.discriminator(estimated_target.detach(), source)

        # Compute discriminative power on fake samples
        target_fake_sample = torch.zeros_like(output_fake_sample)
        loss_fake_sample = self.criterion(output_fake_sample, target_fake_sample)
        disc_loss = loss_real_sample + loss_fake_sample

        # Store generator output to condition next batch on it
        self._store_previous_batch_output(estimated_target, batch_idx)

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

        # If first batch, initialize previous prediction as zero tensor
        if batch_idx == 0:
            self._previous_batch_output = torch.zeros_like(target)

        # Run either generator or discriminator training step
        if optimizer_idx == 0:
            gen_loss, mae = self._step_generator(source, target)
            logs = {'Loss/train_generator': gen_loss,
                    'Metric/train_mae': mae}
            loss = gen_loss + self.l1_weight * mae

        if optimizer_idx == 1:
            disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target, batch_idx)
            logs = {'Loss/train_discriminator': disc_loss,
                    'Metric/train_fooling_rate': fooling_rate,
                    'Metric/train_precision': precision,
                    'Metric/train_recall': recall}
            loss = disc_loss,

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
            source_and_previous_prediction = torch.cat([source, self.logger._previous_batch_output], dim=1)
            output = self(source_and_previous_prediction)

        if self.current_epoch == 0:
            # Log input and groundtruth once only at first epoch
            self.logger.log_images(source[:, :3], tag='Source - Optical (fake RGB)', step=self.current_epoch)
            self.logger.log_images(source[:, -3:], tag='Source - SAR (fake RGB)', step=self.current_epoch)
            self.logger.log_images(target[:, :3], tag='Target - Optical (fake RGB)', step=self.current_epoch)

        # Log generated image and conditioning previous prediction at current epoch
        self.logger.log_images(self.logger._previous_batch_output[:, :3], tag='Previous batch output - Optical (fake RGB)', step=self.current_epoch)
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

        # If first batch, initialize previous prediction as zero tensor
        if batch_idx == 0:
            self._previous_batch_output = torch.zeros_like(target)

        # On next batch, store previous prediction and current batch for image visualization
        if batch_idx == 1:
            self.logger._logging_images = source[:8], target[:8]
            self.logger._previous_batch_output = self._previous_batch_output[:8]

        # Run forward pass on generator and discriminator
        gen_loss, mae = self._step_generator(source, target)
        disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target, batch_idx)

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

        # Make tensorboard logs and return - track discriminator max loss for validation
        logs = {'Loss/val_generator': gen_loss.item(),
                'Loss/val_discriminator': disc_loss.item(),
                'Metric/val_mae': mae.item(),
                'Metric/val_fooling_rate': fooling_rate.item(),
                'Metric/val_precision': precision.item(),
                'Metric/val_recall': recall.item()}
        return {'val_loss': disc_loss, 'log': logs, 'progress_bar': logs}

    def _run_on_single_time_serie(self, source, target):
        """Applies cloud removal to single full time serie tensor - iteratively
        predicts on frame, stores prediction and uses it to condition next
        frame generation

        Args:
            source (torch.Tensor): (horizon, C_source, H, W) input time serie
            target (torch.Tensor): (horizon, C_target, H, W) groundtruth time serie

        Returns:
            type: torch.Tensor
        """
        # Initialize previous prediction to zero tensor
        previous_frame_output = torch.zeros_like(target[0].unsqueeze(0))

        # Apply generator to each time serie frame conditioning on previous prediction
        generated_frames = []
        for frame in source:
            source_and_previous_prediction = torch.cat([frame.unsqueeze(0), previous_frame_output], dim=1)
            generated_frame = self(source_and_previous_prediction)
            generated_frames += [generated_frame]
            previous_frame_output = generated_frame

        # Stack generated frames of time serie into single tensor
        generated_target = torch.cat(generated_frames)
        return generated_target

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

        # Run generator forward pass on single time serie batch
        generated_target = self._run_on_single_time_serie(source, target)

        # Compute performance at downstream classification task
        iou_generated, iou_real = self._compute_legitimacy_at_task_score(self.baseline_classifier,
                                                                         generated_target,
                                                                         target,
                                                                         annotation)

        # Compute IQA metrics
        psnr, ssim, cw_ssim = self._compute_iqa_metrics(generated_target, target)
        mse = F.mse_loss(generated_target, target)
        mae = F.l1_loss(generated_target, target)

        # Encapsulate into torch tensor
        output = torch.Tensor([mae, mse, psnr, ssim, cw_ssim, iou_generated, iou_real])
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
        mae, mse, psnr, ssim, cw_ssim, iou_estimated, iou_real = outputs
        iou_ratio = iou_estimated / iou_real

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
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, horizon):
        self._horizon = horizon
