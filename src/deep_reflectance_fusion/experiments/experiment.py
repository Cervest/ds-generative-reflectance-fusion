import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
import numpy as np
from collections import defaultdict

from src.utils import setseed
from src.deep_reflectance_fusion.evaluation import metrics


class Experiment(pl.LightningModule):
    """General class rehabilitating lightning module logic for reproductible
    experiments.

    Each inheriting classes provide a comprehensive description of an experiment
    with associated model, criterion, optimizer, datasets and execution logic

    LightningModule proposes execution steps to be splitted into "hooks" methods
    provided with a default behavior that can be overwritten.
    Each hook is meant to be called at a specific moment of the execution :

    >>> Epoch training loop roughly runs like :
        ```
        logs = []
        self.on_epoch_start()

        for batch in self.train_dataloader():
            self.on_batch_start(batch)

            loss = self.training_step()
            logs.append(loss)

            self.backward(loss, optimizer)
            self.on_after_backward()

            optimizer.step()
            self.on_before_zero_grad(optimizer)
            optimizer.zero_grad()

            self.on_batch_end()

        self.training_epoch_end(logs)
        self.on_epoch_end()
        ```

    >>> Validation loop roughly runs like :
        ```
        logs = []
        self.on_validation_start()

        for batch in self.val_dataloader():
            self.on_validation_batch_start(batch)

            outs = self.validation_step()
            logs.append(outs)

            self.on_batch_end()

        self.validation_epoch_end(logs)
        ```

    >>> Test loop roughly runs like :
        ```
        logs = []
        self.on_test_start()

        for batch in self.test_dataloader():
            self.on_test_batch_start()

            outs = self.test_step(batch)
            logs.append(outs)

            self.on_test_end()

        self.test_epoch_end(logs)
        ```
    Args:
        model (nn.Module): main model concerned by this experiment
        dataset (torch.utils.data.Dataset): main dataset concerned by this experiment
        split (list[float]): dataset split ratios in [0, 1] as [train, val]
            or [train, val, test]
        dataloader_kwargs (dict): parameters of dataloaders
        optimizer_kwargs (dict): parameters of optimizer defined in LightningModule.configure_optimizers
        lr_scheduler_kwargs (dict): paramters of lr scheduler defined in LightningModule.configure_optimizers
        criterion (nn.Module): differentiable training criterion (default: None)
        seed (int): random seed (default: None)
    """
    def __init__(self, model, dataset, split, dataloader_kwargs, optimizer_kwargs,
                 lr_scheduler_kwargs=None, criterion=None, seed=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.dataloader_kwargs = dataloader_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self._split_and_set_dataset(dataset=dataset,
                                    split=split,
                                    seed=seed)

    @classmethod
    def build(cls, cfg, test=False):
        """Constructor method called on YAML configuration file

        Args:
            cfg (dict): loaded YAML configuration file
            test (bool): set to True for testing

        Returns:
            type: Experiment
        """
        # Build keyed arguments dictionnary out of configurations
        build_kwargs = cls._make_build_kwargs(cfg, test)

        # Instantiate experiment
        if test:
            experiment = cls.load_from_checkpoint(checkpoint_path=cfg['testing']['chkpt'],
                                                  **build_kwargs)
        else:
            if cfg['experiment']['chkpt']:
                experiment = cls.load_from_checkpoint(checkpoint_path=cfg['experiment']['chkpt'],
                                                      **build_kwargs)
            else:
                experiment = cls(**build_kwargs)
        # Set configuration file as hyperparameter
        experiment.hparams = cfg
        return experiment

    @classmethod
    def _make_build_kwargs(cls, cfg, test=False):
        """Build keyed arguments dictionnary out of configurations to be passed
            to class constructor

        This method must be implemented in child class if one wishes using
            YAML class constructor cls.build

        Args:
            cfg (dict): loaded YAML configuration file
            test (bool): set to True for testing

        Returns:
            type: dict
        """
        raise NotImplementedError

    @classmethod
    def _load_model_state(cls, checkpoint, *args, **kwargs):
        """Overrides LightningModule method bypassing the use of hparams dictionnary

        Args:
            checkpoint (dict): Description of parameter `checkpoint`.

        Returns:
            type: Experiment
        """
        # load the state_dict on the model automatically
        model = cls(*args, **kwargs)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model

    @staticmethod
    def _convert_split_ratios_to_length(total_length, split, *args, **kwargs):
        """Convert ratios to lengths - leftovers go to val/test set

        Args:
            total_length (int): total size of dataset
            split (list[float]): dataset split ratios in [0, 1] as [train, val]
                or [train, val, test]

        Returns:
            type: list[int]
        """
        # Convert ratios to lengths - leftovers go to val/test set
        assert sum(split) == 1, f"Split ratios {split} do not sum to 1"
        lengths = [int(r * total_length) for r in split]
        lengths[-1] += total_length - sum(lengths)
        return lengths

    def _random_split(self, dataset, lengths, *args, **kwargs):
        """
        Randomly split a dataset into non-overlapping new datasets of given lengths.

        Args:
            dataset (Dataset): Dataset to be split
            lengths (list[int]): lengths of splits to be produced
        """
        return random_split(dataset, lengths)

    @setseed('torch')
    def _split_and_set_dataset(self, dataset, split, seed=None, *args, **kwargs):
        """Splits dataset into train/val or train/val/test and sets
        splitted datasets as attributes

        Args:
            dataset (torch.utils.data.Dataset)
            split (list[float]): dataset split ratios in [0, 1] as [train, val]
                or [train, val, test]
            seed (int): random seed
        """
        # Convert specified ratios to lengths
        lengths = self._convert_split_ratios_to_length(total_length=len(dataset),
                                                       split=split,
                                                       *args, **kwargs)

        # Split dataset
        datasets = self._random_split(dataset=dataset,
                                      lengths=lengths)

        # Set datasets attributes
        self.train_set = datasets[0]
        self.val_set = datasets[1]
        self.test_set = None if len(datasets) <= 2 else datasets[2]

    @property
    def model(self):
        return self._model

    @property
    def criterion(self):
        return self._criterion

    @property
    def dataloader_kwargs(self):
        return self._dataloader_kwargs

    @property
    def optimizer_kwargs(self):
        return self._optimizer_kwargs

    @property
    def lr_scheduler_kwargs(self):
        return self._lr_scheduler_kwargs

    @property
    def train_set(self):
        return self._train_set

    @property
    def val_set(self):
        return self._val_set

    @property
    def test_set(self):
        return self._test_set

    @model.setter
    def model(self, model):
        self._model = model

    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion

    @dataloader_kwargs.setter
    def dataloader_kwargs(self, dataloader_kwargs):
        self._dataloader_kwargs = dataloader_kwargs

    @optimizer_kwargs.setter
    def optimizer_kwargs(self, optimizer_kwargs):
        self._optimizer_kwargs = optimizer_kwargs

    @lr_scheduler_kwargs.setter
    def lr_scheduler_kwargs(self, lr_scheduler_kwargs):
        self._lr_scheduler_kwargs = lr_scheduler_kwargs

    @train_set.setter
    def train_set(self, train_set):
        self._train_set = train_set

    @val_set.setter
    def val_set(self, val_set):
        self._val_set = val_set

    @test_set.setter
    def test_set(self, test_set):
        self._test_set = test_set


class ImageTranslationExperiment(Experiment):
    """General class factorizing some common attributes and methods of
    image translation experiments
    Args:
        model (nn.Module): main model concerned by this experiment
        dataset (torch.utils.data.Dataset): main dataset concerned by this experiment
        split (list[float]): dataset split ratios in [0, 1] as [train, val]
            or [train, val, test]
        dataloader_kwargs (dict): parameters of dataloaders
        optimizer_kwargs (dict): parameters of optimizer defined in LightningModule.configure_optimizers
        lr_scheduler_kwargs (dict): paramters of lr scheduler defined in LightningModule.configure_optimizers
        criterion (nn.Module): differentiable training criterion (default: None)
        seed (int): random seed (default: None)
    """
    def _compute_classification_metrics(self, output_real_sample, output_fake_sample):
        """Computes metrics on discriminator classification power : fooling rate
            of generator, precision and recall

        Args:
            output_real_sample (torch.Tensor): discriminator prediction on real samples
            output_fake_sample (torch.Tensor): discriminator prediction on fake samples

        Returns:
            type: tuple[float]
        """
        # Setup complete outputs and targets vectors
        target_real_sample = torch.ones_like(output_real_sample)
        target_fake_sample = torch.zeros_like(output_fake_sample)
        output = torch.cat([output_real_sample, output_fake_sample])
        target = torch.cat([target_real_sample, target_fake_sample])

        # Compute generator and discriminator metrics
        fooling_rate = metrics.accuracy(output_fake_sample, target_real_sample)
        precision = metrics.precision(output, target)
        recall = metrics.recall(output, target)
        return fooling_rate, precision, recall

    def _compute_iqa_metrics(self, estimated_target, target):
        """Computes full reference image quality assessment metrics : psnr, ssim
            and spectral angle mapper (see evaluation/metrics/iqa.py for details)

        Args:
            estimated_target (torch.Tensor): generated sample
            target (torch.Tensor): target sample

        Returns:
            type: tuple[float]
        """
        # Reshape as (batch_size * channels, height, width) to run single for loop
        batch_size, channels, height, width = target.shape
        estimated_target = estimated_target.clamp(min=0).detach().cpu().numpy()
        target = target.clamp(min=0).detach().cpu().numpy()
        estimated_bands = estimated_target.reshape(-1, height, width)
        target_bands = target.reshape(-1, height, width)

        # Compute IQA metrics by band
        iqa_metrics = defaultdict(list)
        for src, tgt in zip(estimated_bands, target_bands):
            data_range = np.max([src, tgt])
            src = src / data_range
            tgt = tgt / data_range
            iqa_metrics['psnr'] += [metrics.psnr(tgt, src)]
            iqa_metrics['ssim'] += [metrics.ssim(tgt, src)]

        # Average bandwise
        psnr = np.asarray(iqa_metrics['psnr'])
        ssim = np.asarray(iqa_metrics['ssim'])

        # Compute bandwise average spectral angle mapper
        estimated_target = estimated_target.transpose(0, 2, 3, 1).astype(np.float32)
        target = target.transpose(0, 2, 3, 1).astype(np.float32)
        sam = metrics.sam(target, estimated_target).mean(axis=(0, 1, 2))
        return psnr, ssim, sam
