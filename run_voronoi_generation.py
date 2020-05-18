"""
Runs generation of a toy synthetic imagery product based on voronoi polygons
  (1) Generates segmentation of image as multiple voronoi polygons
  (2) Instantiates product and register polygons
  (3) Generate toy product frames and dump at specified location

Usage: run_generation.py --cfg=<config_file_path>  --o=<output_dir>

Options:
  -h --help             Show help.
  --version             Show version.
  --cfg=<config_path>  Path to config file
  --o=<output_path> Path to output file
"""
from docopt import docopt
import numpy as np
import yaml

from src import PolygonCell, Product, TSDataset, TimeSerie
from src.modules import voronoi


def main(args, cfg):
    # Generate voronoi polygons split of image
    polygons = generate_voronoi_polygons(cfg=cfg)

    # Setup time series dataset
    ts_dataset = make_ts_dataset(cfg=cfg)

    # Instantiate product
    product = make_product(cfg=cfg)

    # Register voronoi polygons
    register_polygons(cfg=cfg,
                      product=product,
                      polygons=polygons,
                      ts_dataset=ts_dataset)

    # Generate and dump product
    product.generate(output_dir=args['--o'], astype=cfg['astype'])


def generate_voronoi_polygons(cfg):
    """Generates n voronoi polygons from random input points
    """
    polygons = voronoi.generate_voronoi_polygons(n=cfg['product']['n_polygons'],
                                                 seed=cfg['seed'])
    return polygons


def make_ts_dataset(cfg):
    """Loads time serie dataset from path specified in cfg
    """
    # Setup time series dataset and artificially keep 3-dims only - to be removed
    ts_dataset = TSDataset(root=cfg['ts_path'])
    ts_dataset._data = ts_dataset._data[['dim_0', 'dim_1', 'dim_2']]
    return ts_dataset


def make_product(cfg):
    """Product initialization adapted to cfg structure
    """
    product_cfg = cfg['product']
    size = product_cfg['size']

    product_kwargs = {'size': (size['width'], size['height']),
                      'nbands': product_cfg['nbands'],
                      'annotation_bands': 2,
                      'horizon': product_cfg['horizon'],
                      'color': product_cfg['background_color'],
                      'seed': cfg['seed']}
    product = Product(**product_kwargs)
    return product


def register_polygons(cfg, product, polygons, ts_dataset):
    """Handles PolygonCell intialization with time serie and registration
    to product
    """
    # Create random filling sampler to break cells homogeneity
    sampler = lambda x: cfg['product']['filling_std'] * np.random.randn(*x)

    for polygon in polygons:
        # Draw random time serie from dataset
        ts_array, ts_label = ts_dataset.choice()

        # Create time serie instance with same or greater horizon
        time_serie = TimeSerie(ts=ts_array, label=ts_label, horizon=product.horizon)

        # Encapsulate at digit level
        cell_kwargs = {'polygon': polygon,
                       'product_size': product.size,
                       'time_serie': time_serie,
                       'sampler': sampler}
        cell = PolygonCell(**cell_kwargs)

        # Register to product
        loc = cell.get_center_loc()
        product.register(cell, loc)


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)
    # Load configuration file
    cfg_path = args["--cfg"]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Run generation
    main(args, cfg)
