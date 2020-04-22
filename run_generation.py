"""Usage: run_generation.py [--cfg=<config_file_path>]  [-o=<output_path>]

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

from src import Digit, Product
from src.utils import list_collate


def main(args, cfg):

    # Setup dataloader
    mnist = MNIST(root="data/mnist", train=True)
    dataloader = DataLoader(dataset=mnist,
                            batch_size=cfg['batch_size'],
                            collate_fn=list_collate)

    # Define digits transforms
    digit_transform = tf.Compose([tf.RandomAffine(degrees=(-90, 90),
                                                  scale=(0.5, 1),
                                                  shear=(-1, 1)),
                                  tf.RandomChoice([tf.RandomHorizontalFlip(0.5),
                                                   tf.RandomVerticalFlip(0.5)]),
                                  tf.RandomPerspective()])
    # Instantiate product
    size = cfg['size']
    grid_size = cfg['grid_size']
    product_kwargs = {'size': (size['width'], size['height']),
                      'mode': cfg['mode'],
                      'grid_size': (grid_size['width'], grid_size['height']),
                      'color': 0,
                      'img_mode': cfg['img_mode'],
                      'blob_transform': digit_transform,
                      'rdm_dist': np.random.randn,
                      'seed': cfg['seed']}

    product = Product(**product_kwargs)

    # Load batch of digits
    digits, labels = iter(dataloader).next()

    # Register to product
    for img, label in zip(digits, labels):
        d = Digit(img, label=label)
        product.random_add(d)

    # Generate and save product
    output = product.generate(seed=cfg['seed'])
    output.save(args['-o'])


if __name__ == "__main__":
    args = docopt(__doc__)
    cfg_path = args["--cfg"]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    main(args, cfg)
