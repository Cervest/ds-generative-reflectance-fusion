"""Usage: run_derivation.py --cfg=<config_file_path>  --o=<output_dir>

Options:
  -h --help             Show help.
  --version             Show version.
  --cfg=<config_path>  Path to config file
  --o=<output_path> Path to output file

 Description: Runs derivation of coarser product by degrading high-resolution one
    (1) Loads toy generated product
    (2) Coarses it through augmentation and downsampling
    (3) Dumps lower resolution product at specified location
"""
from docopt import docopt
import yaml

from src import ProductDataset, Degrader
from src.modules import conv_aggregation, kernels, transforms


def main(args, cfg):

    # Load latent product as product dataset
    latent_dataset = ProductDataset(root=cfg['latent_product_path'])

    # Define augmentation procedure
    transform = transforms.build_transform(cfg['transform'])

    # Define aggregation operator
    heat_kernel = kernels.heat_kernel(size=(4, 4), sigma=1.)
    aggregate_fn = conv_aggregation(heat_kernel)

    # Instantiate degrader
    size = cfg['target_size']
    degrader_kwargs = {'size': (size['width'], size['height']),
                       'temporal_res': cfg['temporal_res'],
                       'transform': transform,
                       'aggregate_fn': aggregate_fn}

    degrader = Degrader(**degrader_kwargs)

    # Derive product from latent dataset
    degrader.derive(product_set=latent_dataset, output_dir=args['--o'])


if __name__ == "__main__":
    args = docopt(__doc__)
    cfg_path = args["--cfg"]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    main(args, cfg)
