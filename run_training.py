"""
Runs training of experiment

Usage: run_training.py --cfg=<config_file_path>  --o=<output_dir> [--device=<execution_device>] [--experiment_name=<name>] [--seed=<random_seed>]

Options:
  -h --help                 Show help.
  --version                 Show version.
  --cfg=<config_file_path>  Path to config file
  --o=<output_directory>    Path to experiments output directory
  --device=<gpus ids>       Ids of GPUs to run training on, None is cpu
  --experiment_name=<name>  Custom naming for subdirectory where experiment outputs are stored
  --seed=<random_seed>      Random seed to use for experiment
"""
import os
from docopt import docopt
import yaml
import pytorch_lightning as pl
from src.rsgan import build_experiment, build_callback
from src.rsgan.experiments import Logger


def main(args, cfg):
    # Set seed for reproducibility
    seed = int(args["--seed"]) if args["--seed"] else cfg['experiment']['seed']
    pl.seed_everything(seed)

    # Build experiment
    experiment = build_experiment(cfg)

    # Build logging and callbacks
    logger = make_logger(args)
    model_checkpoint = make_model_checkpoint(cfg['model_checkpoint'])
    early_stopping = build_callback(cfg['early_stopping'])

    # Instantiate trainer instance
    params = {'logger': logger,
              'early_stop_callback': early_stopping,
              'checkpoint_callback': model_checkpoint,
              'resume_from_checkpoint': cfg['experiment']['chkpt'],
              'precision': cfg['experiment']['precision'],
              'max_epochs': cfg['experiment']['max_epochs'],
              'gpus': args['--device']}
    trainer = pl.Trainer(**params)

    # Run training
    trainer.fit(experiment)


def make_logger(args):
    """Build logger instance pointing to specified output directory
    """
    logger_kwargs = {'save_dir': os.path.dirname(args['--o']),
                     'name': os.path.basename(args['--o'])}

    if args['--experiment_name']:
        logger_kwargs.update({'version': args['--experiment_name']})

    logger = Logger(**logger_kwargs)
    return logger


def make_model_checkpoint(cfg):
    """Build model checkpointing callback
    """
    model_checkpoint = pl.callbacks.ModelCheckpoint(**cfg)
    return model_checkpoint


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load configuration file
    cfg_path = args["--cfg"]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Run generation
    main(args, cfg)
