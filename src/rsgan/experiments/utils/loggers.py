import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid
from src.utils import save_json


class Logger(TensorBoardLogger):
    """Extends pytorch lightning tensorboard logger with image logging method"""

    def __init__(self, save_dir, name="default", version=None, test=False, **kwargs):
        super().__init__(save_dir=save_dir, name=name, version=version, **kwargs)
        self.test = test

    @rank_zero_only
    def log_images(self, images, tag, step):
        self.experiment.add_image(tag=tag,
                                  img_tensor=make_grid(images.cpu(), nrow=8, normalize=True),
                                  global_step=step)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if self.test:
            epoch = metrics['epoch']
            dump_path = os.path.join(self.log_dir, f"test_scores_epoch={epoch}.json")
            save_json(dump_path, metrics)
        else:
            super().log_metrics(metrics, step)

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, test):
        self._test = test
