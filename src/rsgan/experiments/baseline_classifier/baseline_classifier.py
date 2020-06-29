import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.rsgan import build_dataset
from src.rsgan.experiments import EXPERIMENTS
from src.rsgan.experiments.experiment import Experiment
from src.rsgan.evaluation import metrics


@EXPERIMENTS.register('baseline_classifier')
class BaselineClassifier(Experiment):

    def __init__(self, dataset, split, seed=None):
        super().__init__(model=self._make_model(),
                         dataset=dataset,
                         split=split,
                         dataloader_kwargs=None,
                         optimizer_kwargs=None,
                         lr_scheduler_kwargs=None,
                         criterion=nn.BCELoss(),
                         seed=seed)

    def forward(self, x):
        return self.model(x)

    def _make_model(self):
        class TimeSeriesClassifier(nn.Module):
            def __init__(self, ndim):
                super().__init__()
                self.conv_1 = nn.Conv1d(in_channels=ndim, out_channels=64, kernel_size=3, stride=2)
                self.bn_1 = nn.BatchNorm1d(64)
                self.conv_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
                self.bn_2 = nn.BatchNorm1d(32)
                self.conv_3 = nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, stride=2)
                self.out_layer = nn.Linear(3 * 8, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv_1(x)
                x = self.relu(self.bn_1(x))
                x = self.conv_2(x)
                x = self.relu(self.bn_2(x))
                x = self.conv_3(x)
                x = self.out_layer(x.view(x.size(0), -1))
                x = torch.sigmoid(x)
                return x
        return TimeSeriesClassifier(ndim=5)

    def train_dataloader(self):
        """Implements LightningModule test loader building method
        """
        loader = DataLoader(dataset=self.train_set,
                            batch_size=1,
                            num_workers=32)
        return loader

    def val_dataloader(self):
        """Implements LightningModule test loader building method
        """
        loader = DataLoader(dataset=self.val_set,
                            batch_size=1,
                            num_workers=32)
        return loader

    def configure_optimizers(self):
        """Implements LightningModule optimizer and learning rate scheduler
        building method
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        optimizer_dict = {'optimizer': optimizer, 'scheduler': lr_scheduler}
        return optimizer_dict

    def _compute_classification_metrics(self, output, groundtruth):
        accuracy = metrics.accuracy(output, groundtruth)
        precision = metrics.precision(output, groundtruth)
        recall = metrics.recall(output, groundtruth)
        return accuracy, precision, recall

    def training_step(self, batch, batch_idx):
        """Implements LightningModule training logic

        Args:
            batch (tuple[torch.Tensor]): source, target pairs batch
            batch_idx (int)

        Returns:
            type: dict
        """
        # Unfold batch
        time_serie, annotation = batch
        time_serie = time_serie.squeeze()
        annotation = annotation.view(-1, 1)

        # Random indices
        # idx = torch.randperm(len(time_serie))[:256]
        # time_serie = time_serie[idx]
        # annotation = annotation[idx]

        # Run forward step
        prediction = self(time_serie.squeeze())

        # Compute loss
        cross_entropy = self.criterion(prediction, annotation)
        accuracy, precision, recall = self._compute_classification_metrics(prediction, annotation)

        # Setup logs dictionnary
        logs = {'Loss/train_cross_entropy': cross_entropy,
                'Metric/train_accuracy': accuracy,
                'Metric/train_precision': precision,
                'Metric/recall': recall}
        output = {'loss': cross_entropy,
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
        time_serie, annotation = batch
        annotation = annotation.view(-1, 1)
        # Run forward step
        prediction = self(time_serie.squeeze())

        # Compute loss
        cross_entropy = self.criterion(prediction, annotation)
        accuracy, precision, recall = self._compute_classification_metrics(prediction, annotation)

        # Encapsulate scores in torch tensor
        output = torch.Tensor([cross_entropy, accuracy, precision, recall])
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
        cross_entropy, accuracy, precision, recall = outputs

        # Make tensorboard logs and return
        logs = {'Loss/val_cross_entropy': cross_entropy.item(),
                'Metric/val_accuracy': accuracy.item(),
                'Metric/val_precision': precision.item(),
                'Metric/val_recall': recall.item()}
        return {'val_loss': cross_entropy, 'log': logs, 'progress_bar': logs}

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
        build_kwargs = {'dataset': build_dataset(cfg['dataset']),
                        'split': list(cfg['dataset']['split'].values()),
                        'seed': cfg['experiment']['seed']}
        return build_kwargs
