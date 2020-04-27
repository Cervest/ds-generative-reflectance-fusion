"""Usage: run_generation.py --cfg=<config_file_path>  [-o=<output_dir>]

Options:
  -h --help             Show help.
  --version             Show version.
  --cfg=<config_path>  Path to config file
  -o=<output_path> Path to output file [default: /output.png]
"""
from docopt import docopt
import yaml

import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as tf

from src import Digit, Product, TSDataset, TimeSerie
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

    # Define digits transforms
    digit_transform = tf.Compose([tf.RandomAffine(degrees=(-90, 90),
                                                  scale=(0.5, 1),
                                                  shear=(-1, 1)),
                                  tf.RandomChoice([tf.RandomHorizontalFlip(0.5),
                                                   tf.RandomVerticalFlip(0.5)]),
                                  tf.RandomPerspective()])
    # Instantiate product
    product_cfg = cfg['product']
    size = product_cfg['size']
    grid_size = product_cfg['grid_size']
    product_kwargs = {'size': (size['width'], size['height']),
                      'mode': product_cfg['mode'],
                      'nbands': product_cfg['nbands'],
                      'horizon': product_cfg['horizon'],
                      'grid_size': (grid_size['width'], grid_size['height']),
                      'color': 0,
                      'blob_transform': digit_transform,
                      'rdm_dist': np.random.randn,
                      'seed': cfg['seed']}

    product = Product(**product_kwargs)

    # Register batch of digits
    digits, _ = iter(dataloader).next()
    for img in digits:
        ts_array, label = ts_dataset.choice()
        time_serie = TimeSerie(ts_array, label, horizon=product_cfg['horizon'])
        d = Digit(img, label=label, time_serie=time_serie)
        product.random_register(d)

    # Generate and dump product
    product.generate(output_dir=args['-o'], astype='jpg')


if __name__ == "__main__":
    args = docopt(__doc__)
    cfg_path = args["--cfg"]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    main(args, cfg)
