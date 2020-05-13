"""
Runs generation of a toy synthetic imagery product
  (1) Loads MNIST, time serie dataset and setup resizing factors sampler
  (2) Instantiates product and register digits
  (3) Generate toy product frames and dump at specified location

Usage: run_generation.py --cfg=<config_file_path>  --o=<output_dir>

Options:
  -h --help             Show help.
  --version             Show version.
  --cfg=<config_path>  Path to config file
  --o=<output_path> Path to output file
"""
from docopt import docopt
import yaml

import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from src import Digit, Product, TSDataset, TimeSerie, samplers
from src.modules import kernels, transforms
from src.utils import list_collate


def main(args, cfg):
    # Setup mnist dataloader and time series dataset
    mnist_dataloader = make_mnist_dataloader(cfg=cfg)
    ts_dataset = make_ts_dataset(cfg=cfg)

    # Setup covariance function of GP used to update digits sizes
    kernel = kernels.build_kernel(cfg=cfg['scales_sampler']['kernel'])

    # Define digits transforms and instantiate product
    digit_transform = transforms.build_transform(cfg=cfg['product']['transform'])
    product = make_product(cfg=cfg, digit_transform=digit_transform)

    # Register digits to product
    register_digit_batch(cfg=cfg,
                         product=product,
                         mnist_dataloader=mnist_dataloader,
                         ts_dataset=ts_dataset,
                         kernel=kernel)

    # Generate and dump product
    product.generate(output_dir=args['--o'], astype=cfg['astype'])


def make_mnist_dataloader(cfg):
    """Loads MNIST dataset from path specified in cfg and sets up dataloader
    with right batch size
    """
    mnist = MNIST(root=cfg['mnist_path'], train=True)
    dataloader = DataLoader(dataset=mnist,
                            batch_size=cfg['batch_size'],
                            collate_fn=list_collate)
    return dataloader


def make_ts_dataset(cfg):
    """Loads time serie dataset from path specified in cfg
    """
    # Setup time series dataset and artificially keep 3-dims only - to be removed
    ts_dataset = TSDataset(root=cfg['ts_path'])
    ts_dataset._data = ts_dataset._data[['dim_0', 'dim_1', 'dim_2']]
    return ts_dataset


def make_product(cfg, digit_transform):
    """Product initialization adapted to cfg structure
    """
    product_cfg = cfg['product']
    size = product_cfg['size']
    grid_size = product_cfg['grid_size']

    product_kwargs = {'size': (size['width'], size['height']),
                      'mode': product_cfg['mode'],
                      'nbands': product_cfg['nbands'],
                      'annotation_bands': 2,
                      'horizon': product_cfg['horizon'],
                      'grid_size': (grid_size['width'], grid_size['height']),
                      'color': product_cfg['background_color'],
                      'digit_transform': digit_transform,
                      'rdm_dist': np.random.randn,
                      'seed': cfg['seed']}
    product = Product(**product_kwargs)
    return product


def register_digit_batch(cfg, product, mnist_dataloader, ts_dataset, kernel):
    """Handles digits intialization with time serie, scale sampler and
    random registration to product
    """
    # Load batch of digits
    digits, labels = iter(mnist_dataloader).next()

    for img, label in zip(digits, labels):
        # Draw random time serie from dataset
        ts_array, ts_label = ts_dataset.choice()

        # Create time serie instance with same or greater horizon
        time_serie = TimeSerie(ts=ts_array, label=ts_label, horizon=product.horizon)

        # Create GP for scaling digits size with same or greater horizon
        gp_sampler = samplers.ScalingSampler(kernel=kernel, size=product.horizon)

        # Encapsulate at digit level
        digit_kwargs = {'img': img,
                        'label': label,
                        'time_serie': time_serie,
                        'scale_sampler': gp_sampler}
        d = Digit(**digit_kwargs)

        # Register to product
        product.random_register(d)


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)
    # Load configuration file
    cfg_path = args["--cfg"]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Run generation
    main(args, cfg)
