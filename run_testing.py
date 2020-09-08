"""
Runs testing of experiment

Usage: run_testing.py --cfg=<config_file_path>  --o=<output_dir> [--device=<execution_device>] [--chkpt=<path_to_model_checkpoint>]

Options:
  --cfg=<config_file_path>                      Path to config file
  --o=<output_directory>                        Path to output directory
  --device=<gpus_ids>                           Ids of GPUs to run training on, None is cpu
  --chkpt=<path_to_model_checkpoint>            Path to trained model checkpoint to load for evaluation
"""
import os
from docopt import docopt
import pytorch_lightning as pl
from src.deep_reflectance_fusion import build_experiment
from src.deep_reflectance_fusion.experiments import Logger
from src.utils import load_yaml


def main(args, cfg):
    # Build experiment
    experiment = build_experiment(cfg, test=True)

    # Build logging
    logger = make_logger(args, cfg)

    # Instantiate pytorch lightning trainer instance
    params = {'logger': logger,
              'resume_from_checkpoint': cfg['testing']['chkpt'],
              'checkpoint_callback': False,
              'precision': cfg['experiment']['precision'],
              'gpus': args['--device']}
    trainer = pl.Trainer(**params)

    # Run testing
    trainer.test(experiment)


def make_logger(args, cfg):
    """Build logger instance pointing to specified output directory
    """
    save_dir = os.path.dirname(args['--o'])
    name = os.path.basename(args['--o'])
    version = os.path.basename(os.path.dirname(os.path.dirname(cfg['testing']['chkpt'])))
    logger = Logger(save_dir=save_dir, name=name, version=version, test=True)
    return logger


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load configuration file
    cfg = load_yaml(args["--cfg"])

    # Update args if necessary
    if args['--chkpt']:
        cfg['testing']['chkpt'] = args['--chkpt']

    # Run testing
    main(args, cfg)
