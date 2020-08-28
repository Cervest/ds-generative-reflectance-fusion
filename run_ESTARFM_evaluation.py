"""
Description :
    (1) Retrieves testing set correponding to experiment
    (2) Loads ESTARFM predictions and groundtruth from test set
    (3) Compare with full reference image quality metrics
    (4) Dump scores

Usage: run_ESTARFM_evaluation.py --root=<predicted_patches_directory> --target=<groundtruth_patches_directory> --cfg=<experiment_config_file> --o=<output_dir>

Options:
  --root=<predicted_patches_directory>                  ESTARFM predicted patches directory
  --target=<groundtruth_patches_directory>              Groundtruth patches directory
  --cfg=<experiment_config_file>                        Experiment configuration file containing dataset split
  --o=<output_dir>                                      Output directory
"""
import os
from collections import defaultdict
from docopt import docopt
import logging
import numpy as np
import rasterio
from progress.bar import Bar

from src.rsgan.experiments import build_experiment
from src.rsgan.evaluation import metrics
from src.utils import load_yaml, save_json


def main(args):
    root = args['--root']
    experiment = build_experiment(load_yaml(args['--cfg']))
    bar = Bar("Patch directory", max=len(experiment.test_set))
    iqa_metrics = defaultdict(list)

    for patch_idx in patches_subset_from(experiment.test_set):
        patch_directory = os.path.join(root, patch_idx)
        if not os.path.isdir(patch_directory):
            # Some patches aren't predicted by ESTARFM as it requires a sample before and one after
            continue

        for date in os.listdir(patch_directory):
            # Load predicted bands
            date_directory = os.path.join(patch_directory, date)
            files_paths = [os.path.join(date_directory, band) for band in os.listdir(date_directory)]
            predicted_bands = load_in_multiband_raster(files_paths)

            # Load groundtruth bands
            target_directory = os.path.join(args['--target'], patch_idx, 'landsat', date)
            target_files_paths = [os.path.join(target_directory, band) for band in os.listdir(target_directory)]
            target_bands = load_in_multiband_raster(target_files_paths)

            # Compute PSNR and SSIM by band
            patch_bands_iqa = defaultdict(list)
            for src, tgt in zip(predicted_bands, target_bands):
                data_range = np.max([src, tgt])
                src = src.clip(min=np.finfo(np.float16).eps) / data_range
                tgt = tgt.clip(min=np.finfo(np.float16).eps) / data_range
                patch_bands_iqa['psnr'] += [metrics.psnr(tgt, src)]
                patch_bands_iqa['ssim'] += [metrics.ssim(tgt, src)]

            # Record bandwise value
            iqa_metrics['psnr'] += [patch_bands_iqa['psnr']]
            iqa_metrics['ssim'] += [patch_bands_iqa['ssim']]

            # Compute bandwise spectral angle mapper
            predicted_patch = np.dstack(predicted_bands).astype(np.float32)
            target_patch = np.dstack(target_bands).astype(np.float32)
            sam = metrics.sam(target_patch, predicted_patch).mean(axis=(0, 1))
            iqa_metrics['sam'] += [sam]

        # Log running averages
        avg_psnr, avg_ssim, avg_sam = np.mean(iqa_metrics['psnr']), np.mean(iqa_metrics['ssim']), np.mean(iqa_metrics['sam'])
        bar.suffix = "PSNR = {:.2f} | SSIM = {:.4f} | SAM = {:.6f}".format(avg_psnr, avg_ssim, avg_sam)
        bar.next()

    # Make bandwise average output dictionnary
    bandwise_avg_psnr = np.asarray(iqa_metrics['psnr']).mean(axis=0).astype(np.float64)
    bandwise_avg_ssim = np.asarray(iqa_metrics['ssim']).mean(axis=0).astype(np.float64)
    bandwise_avg_sam = np.asarray(iqa_metrics['sam']).mean(axis=0).astype(np.float64)

    avg_iqa_metrics = {'test_psnr': bandwise_avg_psnr.tolist(),
                       'test_ssim': bandwise_avg_ssim.tolist(),
                       'test_sam': bandwise_avg_sam.tolist()}
    os.makedirs(args['--o'], exist_ok=True)
    dump_path = os.path.join(args['--o'], f"test_scores_starfm.json")
    save_json(dump_path, avg_iqa_metrics)


def patches_subset_from(subset):
    """Retrieves list of patch directory names used in dataset subset
    """
    f = lambda idx: os.path.basename(subset.dataset[idx].root)
    return map(f, subset.indices)


def load_in_multiband_raster(files_paths):
    """Loads in list of rasters loaded from specified files paths
    """
    raster_files_path = filter(lambda x: x.endswith('.tif'), files_paths)
    return [rasterio.open(path, 'r').read(1) for path in raster_files_path]


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')

    # Run generation
    main(args)
