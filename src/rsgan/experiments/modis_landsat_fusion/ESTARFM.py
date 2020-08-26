"""
Description : Runs ESTARFM to perform MODIS-Landsat surface reflectance fusion

Usage: ESTARFM.py --root=<patches_directory>

Options:
  -h --help                                             Show help.
  --version                                             Show version.
  --root=<output_directory>                             Input patches directory
"""
import os
import subprocess
from docopt import docopt
import logging
from progress.bar import Bar


def main(args):
    root = args['--root']
    patch_directories = [os.path.join(root, x) for x in os.listdir(root)]
    bar = Bar("Patch directory", max=len(patch_directories))

    for patch_directory in patch_directories:
        params_directory = os.path.join(patch_directory, 'params')
        params_paths = [os.path.join(params_directory, x) for x in os.listdir(params_directory)]

        for param_path in params_paths:
            cmd = f'../cuESTARFM/Codes/cuESTARFM {param_path}'
            subprocess.check_output(cmd, shell=True)

        bar.next()


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')

    # Run generation
    main(args)
