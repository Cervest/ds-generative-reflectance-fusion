"""Usage: run_generation.py [--config=<cache_path>]  [-o=<output_path>]

Options:
  -h --help             Show help.
  --version             Show version.
  --config=<config_path>  Path to config file
  -o=<output_path> Path to output file [default: /output.png]
"""
import os
import sys
import yaml
from PIL import Image
import numpy as np
from docopt import docopt

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as tf

from src import Digit, Product
from src.utils import list_collate


def main(args, config):

    # Setup dataloader
    mnist = MNIST(root="data/mnist", train=True)
    dataloader = DataLoader(dataset=mnist,
                            batch_size=config['batch_size'],
                            collate_fn=list_collate)

    # Define digits transforms
    digit_transform = tf.Compose([tf.RandomAffine(degrees=(-90, 90),
                                                  scale=(0.5, 1),
                                                  shear=(-1, 1)),
                                  tf.RandomChoice([tf.RandomHorizontalFlip(0.5),
                                                   tf.RandomVerticalFlip(0.5)]),
                                  tf.RandomPerspective()])
    # Instantiate product
    product_kwargs = {'size': (300, 300),
                      'color': 0,
                      'mode': 'L',
                      'blob_transform': digit_transform}

    product = Product(**product_kwargs)

    # Load batch of digits
    digits, labels = iter(dataloader).next()

    # Register to product
    for img, label in zip(digits, labels):
        d = Digit(img, label=label)
        product.random_add(d)

    # Generate and save product
    output = product.generate(seed=config['seed'])
    output.save(args['-o'])


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    config_path = args["--config"]
    config = yaml.load(config_path)
    main(args, config)
