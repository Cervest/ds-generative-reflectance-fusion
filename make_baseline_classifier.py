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
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.rsgan import build_experiment
from src.toygeneration import ProductDataset
from src.utils import save_pickle, setseed


def main(args, cfg):
    # Build experiment
    experiment = build_experiment(cfg)

    # Retrieve dataloaders of annotated target frames from training and validation set
    logging.info("Loading training and validation sets")
    train_loader, val_loader = make_annotated_clean_frames_dataloaders(experiment)

    # Convert into (n_pixel, n_channel), (n_pixel,) arrays for sklearn
    logging.info("Converting datasets to arrays")
    X_train, y_train = dataset_as_arrays(train_loader, seed=cfg['experiment']['seed'])
    X_val, y_val = dataset_as_arrays(val_loader, seed=cfg['experiment']['seed'])

    # Remove background pixels which we are not interested in classifying
    X_train, y_train = filter_background_pixels(X_train, y_train)
    X_val, y_val = filter_background_pixels(X_train, y_train)

    # Fit random forest classifier to training set
    classifier_cfg = cfg['baseline_classifier']
    rf = fit_random_forest_classifier_by_chunks(X_train=X_train, y_train=y_train,
                                                n_estimators=classifier_cfg['n_estimators'],
                                                n_chunks=classifier_cfg['n_chunks'],
                                                min_samples_split=classifier_cfg['min_samples_split'],
                                                max_depth=classifier_cfg['max_depth'],
                                                min_samples_leaf=classifier_cfg['min_samples_leaf'],
                                                seed=classifier_cfg['seed'],
                                                n_jobs=int(args['--njobs']))

    # Dump classifier at specified location
    dump_path = args['--o']
    logging.info(f"Saving classifier at {dump_path}")
    save_pickle(dump_path, rf)

    # Compute and save validation accuracy
    logging.info("Computing accuracy on validation set")
    compute_and_save_validation_accuracy(X_val, y_val, rf, dump_path)

    # Compute and save confusion matric on validation set
    logging.info("Computing confusion matrix on validation set")
    compute_and_save_confusion_matrix(X_val, y_val, rf, dump_path)


def make_annotated_clean_frames_dataloaders(experiment):
    """Builds train and validation dataloader of clean groundtruth frames along
    with their pixel-label masks. Respects experiment train/val split to avoid
    training on validation and testing data

    Applies same normalization procedure to frames than the one used for
    training generative models
    """
    # Retrieve clean groundtruth frames dataset which are targets in generative models training
    enhanced_annotated_frames_dataset = experiment.train_set.dataset.enhanced_optical_dataset

    # Set normalization transform for frames
    set_transform_recursively(concat_dataset=enhanced_annotated_frames_dataset,
                              transform=lambda x: (x - 0.5) / 0.5,
                              attribute_name='frame_transform')

    # Set pixel-label selection transform for annotation masks
    set_transform_recursively(concat_dataset=enhanced_annotated_frames_dataset,
                              transform=lambda x: x[:, :, 1],
                              attribute_name='annotation_transform')

    # Build dataloaders restricted to corresponding indices sets
    train_indices = experiment.train_set.indices
    train_loader = make_random_subset_dataloader_from_indices(dataset=enhanced_annotated_frames_dataset,
                                                              full_indices=train_indices,
                                                              size=len(train_indices) // 10)
    val_indices = experiment.val_set.indices
    val_loader = make_random_subset_dataloader_from_indices(dataset=enhanced_annotated_frames_dataset,
                                                            full_indices=val_indices,
                                                            size=len(val_indices) // 10)
    return train_loader, val_loader


def set_transform_recursively(concat_dataset, transform, attribute_name):
    """Used datasets results of recursive concatenation of ProductDataset
    instances encapsulated under torch.data.utils.ConcatDataset instances as a
    binary tree :

                            +---+ConcatDataset+---+
                            |                     |
                            +                     +
                 +---+ConcatDataset+---+    ProductDataset
                 |                     |
                 +                     +
           ConcatDataset         ProductDataset
                 +
                 |
                ...

    However, we want to define transforms for the leaves ProductDataset instances,
    we hence need to operate recursively

    Args:
        concat_dataset (torch.utils.data.ConcatDataset)
        transform (callable): np.ndarray -> np.ndarray
        attribute_name (str): attribute name where to set transform
    """
    concat_dataset.datasets[1].__setattr__(attribute_name, transform)
    if isinstance(concat_dataset.datasets[0], ProductDataset):
        concat_dataset.datasets[0].__setattr__(attribute_name, transform)
    else:
        set_transform_recursively(concat_dataset.datasets[0], transform, attribute_name)


def make_random_subset_dataloader_from_indices(dataset, full_indices, size):
    """Builds dataloader from some dataset on a random subset of specified size
    drawn out of specified indices

    Args:
        dataset (torch.utils.data.Dataset)
        full_indices (list[int]): list of allowed dataset indices
        size (int): number of indices to randomly draw to build dataloader

    Returns:
        type: torch.utils.data.DataLoader
    """
    indices = np.random.choice(full_indices, size=size, replace=False)
    dataloader = DataLoader(dataset=dataset,
                            sampler=SubsetRandomSampler(indices=indices))
    return dataloader


@setseed('numpy')
def dataset_as_arrays(dataloader, seed):
    """Drain out dataloader to load frames and labels into memory as numpy arrays,
    ready to be fed to random forest classifier
    """
    # Unpack frames and annotations
    frames, annotations = list(zip(*list(dataloader)))

    # Reshape such that each pixel is a sample and channels features + convert to numpy
    X = torch.cat(frames)
    X = X.view(-1, X.size(-1)).numpy()
    y = torch.cat(annotations).flatten().numpy()

    # Shuffle jointly pixel and labels
    shuffled_indices = np.random.permutation(len(X))
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    return X, y


def filter_background_pixels(X, y):
    foreground_pixels = y != 0
    return X[foreground_pixels], y[foreground_pixels]


def fit_random_forest_classifier_by_chunks(X_train, y_train,
                                           n_estimators, n_chunks,
                                           min_samples_split, max_depth,
                                           min_samples_leaf, seed, n_jobs):
    """Fits classifier by chunk as dataset is to big to be fitted at once.
    Downside is that no partial fit option is provided with RandomForestClassifier
    and only option is to add new estimators for each new chunk and fit them while
    previously fitted estimators don't change
    """
    # Compute number of estimators to add per chunk
    n_estimators_by_chunk = n_estimators // n_chunks

    # Instantiate random forest classifier with no estimator
    rf_kwargs = {'n_estimators': 0,
                 'max_features': 'auto',
                 'min_samples_split': min_samples_split,
                 'n_jobs': n_jobs,
                 'max_depth': max_depth,
                 'min_samples_leaf': min_samples_leaf,
                 'warm_start': True,
                 'random_state': seed,
                 'class_weight': 'balanced'}
    rf = RandomForestClassifier(**rf_kwargs)
    logging.info(rf)

    # Fit to training dataset by chunks
    chunks_iterator = zip(np.array_split(X_train, n_chunks), np.array_split(y_train, n_chunks))
    for i, (chunk_X, chunk_y) in enumerate(chunks_iterator):
        logging.info(f"Fitting random forest classifier on {len(chunk_X)} pixels")
        rf.n_estimators += n_estimators_by_chunk
        rf.fit(chunk_X, chunk_y)
    return rf


def compute_and_save_validation_accuracy(X_val, y_val, classifier, dump_path):
    val_accuracy = classifier.score(X_val, y_val)
    val_accuracy_dump_path = os.path.join(os.path.dirname(dump_path), "val_accuracy.metric")
    with open(val_accuracy_dump_path, 'w') as f:
        f.write(str(val_accuracy))
    logging.info(f"Validation accuracy : {val_accuracy} - dumped at {val_accuracy_dump_path}")


def compute_and_save_confusion_matrix(X_val, y_val, classifier, dump_path):
    y_pred = classifier.predict(X_val)
    cm = confusion_matrix(y_val, y_pred, labels=classifier.classes_, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classifier.classes_)
    confusion_matrix_dump_path = os.path.join(os.path.dirname(dump_path), "confusion_matrix.png")
    save_confusion_matrix_plot(disp, confusion_matrix_dump_path)
    logging.info(f"Confusion matrix saved at {confusion_matrix_dump_path}")


def save_confusion_matrix_plot(disp, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(include_values=True,
              cmap='magma',
              ax=ax)
    plt.tight_layout()
    plt.savefig(path)


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
