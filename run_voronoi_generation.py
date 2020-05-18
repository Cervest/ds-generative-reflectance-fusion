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
from scipy import stats
import yaml
import logging

from src import PolygonCell, Product, TSDataset, TimeSerie
from src.modules import GPSampler, voronoi, kernels
from src.timeserie import utils as ts_utils


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
    n_polygons = cfg['product']['n_polygons']
    seed = cfg['seed']
    logging.info(f"Generating {n_polygons} polygons from random seed {seed}")

    polygons = voronoi.generate_voronoi_polygons(n=n_polygons,
                                                 seed=seed)
    return polygons


def make_ts_dataset(cfg):
    """Loads time serie dataset from path specified in cfg
    """
    ts_cfg = cfg['ts']

    # Setup TS dataset and artificially keep nb of dims and labels specified
    ts_dataset = TSDataset(root=ts_cfg['path'])
    ts_dataset = ts_utils.truncate_dimensions(ts_dataset, ndim=ts_cfg['ndim'])
    ts_dataset = ts_utils.group_labels(ts_dataset, n_groups=ts_cfg['nlabels'])
    ts_dataset = ts_utils.min_max_rescale(ts_dataset)

    # Draw list of labels for polygons according to label distribution
    labels_dist = ts_utils.discretize_over_points(stats_dist=stats.expon,
                                                  n_points=len(np.unique(ts_dataset.labels)))
    ts_dataset._draw_label_list(size=cfg['product']['n_polygons'],
                                distribution=labels_dist,
                                seed=cfg['seed'])
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


def make_random_sampler(cfg):
    """Build random sampler callable used to break polygons filling homogeneity
    """
    sampler_cfg = cfg['random_sampler']
    if sampler_cfg['name'] == 'gaussian_process':
        sampler = GPSampler(mean=lambda x: np.zeros(x.shape[0]),
                            kernel_name=sampler_cfg['kernel']['name'])
    elif sampler_cfg['name'] == 'gaussian':
        std = sampler_cfg['std']
        sampler = lambda size: std * np.random.randn(*size)
    else:
        raise ValueError("Invalid random sampler name specified")
    return sampler


def compute_cholesky_decomposition(cfg, product, polygons):
    kernel_cfg = cfg['random_sampler']['kernel']
    kernel = kernels.build_kernel(cfg=kernel_cfg)
    size_max = np.max([PolygonCell.img_size_from_polygon(p, product.size) for p in polygons])

    logging.info(f'Computing Cholesky decomposition of ({size_max},{size_max}) covariance matrix')
    GPSampler._cache_cholesky(name=kernel_cfg['name'],
                              size=(size_max, size_max),
                              kernel=kernel)


def register_polygons(cfg, product, polygons, ts_dataset):
    """Handles PolygonCell intialization with time serie and registration
    to product
    """
    # Compute and cache choleski decomposition from largest polygon size
    if cfg['random_sampler']['name'] == 'gaussian_process':
        compute_cholesky_decomposition(cfg, product, polygons)

    # Get polygons label sequence
    label_sequence = ts_dataset._labels_order_list
    logging.info(f"Registering polygons with label sequence {label_sequence[:30]}...")

    for polygon, label in zip(polygons, label_sequence):
        # Draw random time serie from dataset
        ts_array, ts_label = ts_dataset.choice(label=label)

        # Create time serie instance with same or greater horizon
        time_serie = TimeSerie(ts=ts_array, label=ts_label, horizon=product.horizon)

        # Create sampler instance
        sampler = make_random_sampler(cfg)

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
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')
    # Load configuration file
    cfg_path = args["--cfg"]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Run generation
    main(args, cfg)
