from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid


class Logger(TensorBoardLogger):
    """Extends pytorch lightning tensorboard logger with image logging method"""

    @rank_zero_only
    def log_images(self, images, tag, step):
        self.experiment.add_image(tag=tag,
                                  img_tensor=make_grid(images.cpu(), nrow=8, normalize=True),
                                  global_step=step)
