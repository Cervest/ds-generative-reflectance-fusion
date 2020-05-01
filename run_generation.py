"""Usage: run_generation.py --cfg=<config_file_path>  --o=<output_dir>

Options:
  -h --help             Show help.
  --version             Show version.
  --cfg=<config_path>  Path to config file
  --o=<output_path> Path to output file

Description: Runs generation of a toy synthetic imagery product
  (1) Loads MNIST, time serie dataset and setup resizing factors sampler
  (2) Instantiates product and register digits
  (3) Generate toy product frames and dump at specified location
"""
from docopt import docopt
import yaml

import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as tf

from src import Digit, Product, TSDataset, TimeSerie, transforms, samplers
from src.modules import kernels
from src.utils import list_collate


def main(args, cfg):

    # Setup mnist dataloader
    mnist = MNIST(root=cfg['mnist_path'], train=True)
    dataloader = DataLoader(dataset=mnist,
                            batch_size=cfg['batch_size'],
                            collate_fn=list_collate)

    # Setup time series dataset and artificially keep 3-dims only
    ts_dataset = TSDataset(root=cfg['ts_path'])
    ts_dataset._data = ts_dataset._data[['dim_0', 'dim_1', 'dim_2']]

    # Setup mean and covariance of GP used to update digits sizes
    mean = np.zeros_like
    kernel = kernels.rbf(sigma=2.)

    # Define digits transforms
    digit_transform = tf.Compose([tf.RandomAffine(degrees=(-90, 90),
                                                  scale=(0.5, 1),
                                                  shear=(-1, 1)),
                                  tf.RandomChoice([tf.RandomHorizontalFlip(0.5),
                                                   tf.RandomVerticalFlip(0.5)]),
                                  tf.RandomPerspective(),
                                  transforms.RandomScale(scale=(5, 15))])
    # Instantiate product
    product_cfg = cfg['product']
    size = product_cfg['size']
    grid_size = product_cfg['grid_size']
    product_kwargs = {'size': (size['width'], size['height']),
                      'mode': product_cfg['mode'],
                      'nbands': product_cfg['nbands'],
                      'horizon': product_cfg['horizon'],
                      'grid_size': (grid_size['width'], grid_size['height']),
                      'color': product_cfg['background_color'],
                      'blob_transform': digit_transform,
                      'rdm_dist': np.random.randn,
                      'seed': cfg['seed']}

    product = Product(**product_kwargs)

    # Register batch of digits
    digits, labels = iter(dataloader).next()
    for img, label in zip(digits, labels):
        # Draw random time serie from dataset
        ts_array, ts_label = ts_dataset.choice()
        # Create time serie instance with same or greater horizon
        time_serie = TimeSerie(ts=ts_array, label=ts_label, horizon=product_cfg['horizon'])
        # Create GP for scaling digits size with same or greater horizon
        gp_sampler = samplers.ScalingSampler(mean=mean, kernel=kernel, size=product.horizon)
        # Encapsulate at digit level
        digit_kwargs = {'img': img,
                        'label': label,
                        'time_serie': time_serie,
                        'scale_sampler': gp_sampler}

        d = Digit(**digit_kwargs)
        product.random_register(d)

    # Generate and dump product
    product.generate(output_dir=args['--o'], astype=cfg['astype'])


if __name__ == "__main__":
    args = docopt(__doc__)
    cfg_path = args["--cfg"]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    main(args, cfg)
