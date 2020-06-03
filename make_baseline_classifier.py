"""
Runs random forest pixel time series classifier training

Usage: make_baseline_classifier.py --cfg=<config_file_path>  --o=<output_dir> [--njobs=<number_of_workers>]

Options:
  -h --help             Show help.
  --version             Show version.
  --cfg=<config_file_path>  Path to config file
  --o=<output_directory> Path to output directory
  --njobs=<number_of_workers> Number of workers for parallelization [default: -1]
"""
import os
from docopt import docopt
import logging
import yaml
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.ensemble import RandomForestClassifier
from src.rsgan import build_experiment
from src.utils import save_pickle


def main(args, cfg):
    # Build experiment
    experiment = build_experiment(cfg)

    # Retrieve dataloaders of annotated target frames from training and validation set
    logging.info("Loading training and validation sets")
    train_loader, val_loader = get_annotated_dataloaders(experiment)

    # Convert into (n_pixel, n_channel), (n_pixel,) arrays for sklearn
    logging.info("Converting datasets to arrays")
    X_train, y_train = dataset_as_arrays(train_loader)
    X_val, y_val = dataset_as_arrays(val_loader)

    # Make Random Forest classifier
    rf_kwargs = {'n_estimators': 100,
                 'max_features': 'auto',
                 'min_samples_split': 2,
                 'n_jobs': cfg['--njobs'],
                 'random_state': 42}
    rf = RandomForestClassifier(**rf_kwargs)

    # Fit to training dataset
    logging.info(f"Fitting random forest classifier on {len(train_loader)} frames \n {rf}")
    rf.fit(X_train, y_train)

    # Dump classifier at specified location
    dump_path = args['--o']
    logging.info(f"Saving classifier at {dump_path}")
    save_pickle(dump_path, rf)

    # Compute and save validation accuracy
    logging.info("Computing accuracy on validation set")
    val_score = rf.score(X_val, y_val)
    val_score_dump_path = os.path.join(os.path.dirname(dump_path), "val_score.metric")
    with open(val_score_dump_path, 'w') as f:
        f.write(str(val_score))
    logging.info(f"Validation accuracy : {val_score} - dumped at {val_score_dump_path}")


def get_annotated_dataloaders(experiment):
    target_dataset = experiment.train_set.dataset.enhanced_optical_dataset
    # train_indices = experiment.train_set.indices
    train_indices = [73, 83]
    train_loader = DataLoader(dataset=target_dataset,
                              sampler=SubsetRandomSampler(indices=train_indices))

    # val_indices = experiment.val_set.indices
    val_indices = [5, 42, 23]
    val_loader = DataLoader(dataset=target_dataset,
                            sampler=SubsetRandomSampler(indices=val_indices))
    return train_loader, val_loader


def dataset_as_arrays(dataloader):
    # Unpack frames and annotations
    frames, annotations = list(zip(*list(dataloader)))

    # Reshape such that each pixel is a sample and channels features
    X = torch.cat(frames)
    X = X.view(-1, X.size(-1)).numpy()

    # Keep only second layer of annotations - corresponds to time series labels
    y = torch.cat(annotations)[:, :, :, 1].flatten().numpy()
    return X, y


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
