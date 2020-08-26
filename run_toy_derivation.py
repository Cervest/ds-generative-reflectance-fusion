"""
Runs derivation of coarser product by degrading a high-resolution one
  (1) Loads toy generated product
  (2) Coarses it through augmentation and downsampling
  (3) Dumps lower resolution product at specified location

Usage: run_derivation.py --cfg=<config_file_path>  --o=<output_dir> [--product=<path_to_latent_product>]

Options:
  -h --help                             Show help.
  --version                             Show version.
  --cfg=<config_path>                   Path to config file
  --o=<output_path>                     Path to output file
  --product=<path_to_latent_product>    Path to latent product to derive
"""
from docopt import docopt
import yaml

from src.toygeneration import ProductDataset, Degrader
from src.toygeneration.modules import conv_aggregation, kernels, transforms


def main(args, cfg):

    # Load latent product as product dataset
    latent_dataset = load_product_dataset(cfg=cfg)

    # Define augmentation procedure
    corruption_transform = transforms.build_transform(cfg=cfg['corruption'])
    geometric_transform = transforms.build_transform(cfg=cfg['deformation'])
    postprocess_transform = transforms.build_transform(cfg=cfg['postprocess'])

    # Define aggregation operator
    aggregate_fn = make_aggregation_operator(cfg=cfg)

    # Instantiate degrader
    degrader = make_degrader(cfg=cfg,
                             corruption_transform=corruption_transform,
                             geometric_transform=geometric_transform,
                             postprocess_transform=postprocess_transform,
                             aggregate_fn=aggregate_fn)

    # Derive product from latent dataset
    degrader.derive(product_set=latent_dataset, output_dir=args['--o'])


def load_product_dataset(cfg):
    """Loads latent product to derive as a product dataset
    """
    latent_dataset = ProductDataset(root=cfg['latent_product_path'])
    return latent_dataset


def make_aggregation_operator(cfg):
    """Builds heat kernel given cfg specification and derives aggregation
    callable
    """
    if cfg['aggregation']:
        # Compute kernel dimensions
        cfg_kernel = cfg['aggregation']['kernel']
        latent_size = cfg['aggregation']['latent_size']
        target_size = cfg['target_size']
        kernel_width = latent_size['width'] // target_size['width']
        kernel_height = latent_size['height'] // target_size['height']

        # Build aggregation operator
        heat_kernel = kernels.heat_kernel(size=(kernel_width, kernel_height),
                                          sigma=cfg_kernel['sigma'])
        aggregate_fn = conv_aggregation(heat_kernel)
        return aggregate_fn


def make_degrader(cfg, corruption_transform, geometric_transform, postprocess_transform, aggregate_fn):
    """Degrader initialization adapted to cfg structure
    """
    size = cfg['target_size']

    degrader_kwargs = {'size': (size['width'], size['height']),
                       'temporal_res': cfg['temporal_res'],
                       'corruption_transform': corruption_transform,
                       'geometric_transform': geometric_transform,
                       'postprocess_transform': postprocess_transform,
                       'aggregate_fn': aggregate_fn}
    degrader = Degrader(**degrader_kwargs)
    return degrader


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load configuration file
    cfg_path = args["--cfg"]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Update latent product to derive if specified
    if args['--product']:
        cfg.update({'latent_product_path': args['--product']})
    # Run derivation
    main(args, cfg)
