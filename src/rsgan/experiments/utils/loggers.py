import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid
from src.utils import save_json


class Logger(TensorBoardLogger):
    """Extends pytorch lightning tensorboard logger with :
        - Image logging method
        - Separate logging directories in training and testing modes -> useful for dvc outputs
        - Custom logging method for metrics in testing mode

    Args:
        save_dir (str): name of root directory to save experiment logs into
        name (str): optional subdirectory to save_dir
        version (str): name of subdirectory for experiment trial version
        test (bool): if True, logging in testing mode
    """
    _train_subdirectory_name = "run"
    _test_subdirectory_name = "eval"

    def __init__(self, save_dir, name=None, version=None, test=False, **kwargs):
        super().__init__(save_dir=save_dir, name=name, version=version, **kwargs)
        self.test = test

    @rank_zero_only
    def log_images(self, images, tag, step):
        self.experiment.add_image(tag=tag,
                                  img_tensor=make_grid(images.cpu(), nrow=8, normalize=True),
                                  global_step=step)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        # If on testing mode, log output score as json file
        if self.test:
            epoch = metrics['epoch']
            dump_path = os.path.join(self.log_dir, f"test_scores_epoch={epoch}.json")
            save_json(dump_path, metrics)
        # Else, usual tensorboard logging mode
        else:
            super().log_metrics(metrics, step)

    @property
    def log_dir(self):
        """
        On training mode : logdir = save_dir/version/run
        On testing mode : logdir = save_dir/version/eval
        """
        log_dir = os.path.join(super().log_dir, self.subdirectory_name)
        return log_dir

    @property
    def test(self):
        return self._test

    @property
    def subdirectory_name(self):
        return self._test_subdirectory_name if self.test else self._train_subdirectory_name

    @test.setter
    def test(self, test):
        self._test = test
