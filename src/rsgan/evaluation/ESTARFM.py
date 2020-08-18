"""
Description : Iterates over paired patch arrays extracted from Landsat and MODIS data
    and splits them into individual raster by band in a structured directory

Usage: evaluate_ESTARFM.py --root=<predicted_patch_directory> --target=<groundtruth_patch_directory> --o=<output_dir> --cfg=<config_file>

Options:
  -h --help                                             Show help.
  --version                                             Show version.
  --root=<output_directory>                             Input patches directory
"""
import os
import sys
from collections import defaultdict
from docopt import docopt
import logging
import numpy as np
import rasterio
from progress.bar import Bar


base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../")
sys.path.append(base_dir)

from src.rsgan.experiments import build_experiment
from src.rsgan.evaluation import metrics
from src.utils import load_yaml, save_json


def main(args):
    root = args['--root']
    experiment = build_experiment(load_yaml(args['--cfg']))
    bar = Bar("Patch directory", max=len(os.listdir(root)))
    iqa_metrics = defaultdict(list)

    for patch_idx in patches_subset_from(experiment.test_set):
        patch_directory = os.path.join(root, patch_idx)
        for date in os.listdir(patch_directory):
            # Load predicted frame
            date_directory = os.path.join(patch_directory, date)
            files_paths = [os.path.join(date_directory, band) for band in os.listdir(date_directory)]
            predicted_bands = load_in_multiband_raster(files_paths)

            # Load groundtruth frame
            target_directory = os.path.join(args['--target'], patch_idx, date, 'landsat')
            target_files_paths = [os.path.join(date_directory, band) for band in os.listdir(target_directory)]
            target_bands = load_in_multiband_raster(target_files_paths)

            # Compute PSNR and SSIM by band
            patch_iqa_metrics = defaultdict(list)
            for src, tgt in zip(predicted_bands, target_bands):
                data_range = np.max([src, tgt])
                src = src / data_range
                tgt = tgt / data_range
                patch_iqa_metrics['psnr'] += [metrics.psnr(tgt, src)]
                patch_iqa_metrics['ssim'] += [metrics.ssim(tgt, src)]

            # Record band-average value
            iqa_metrics['psnr'] += [np.mean(patch_iqa_metrics['psnr'])]
            iqa_metrics['ssim'] += [np.mean(patch_iqa_metrics['ssim'])]

            # Compute average SAM
            predicted_patch = np.dstack(predicted_bands).astype(np.float32)
            target_patch = np.dstack(target_bands).astype(np.float32)
            sam = metrics.sam(target_patch, predicted_patch, reduce='mean')
            iqa_metrics['sam'] += [sam]

        avg_psnr, avg_ssim, avg_sam = np.mean(iqa_metrics['psnr']), np.mean(iqa_metrics['ssim']), np.mean(iqa_metrics['sam'])
        bar.suffix = "PSNR = {:.2f} | SSIM = {:.4f} | SAM = {:.6f}".format(avg_psnr, avg_ssim, avg_sam)
        bar.next()

    avg_iqa_metrics = {'psnr': avg_psnr, 'ssim': avg_ssim, 'sam': avg_sam}
    dump_path = os.path.join(args['--o'], f"test_scores_starfm.json")
    save_json(dump_path, avg_iqa_metrics)


def patches_subset_from(subset):
    f = lambda idx: os.path.basename(subset.dataset[idx].root)
    return map(f, subset.indices)


def load_in_multiband_raster(files_paths):
    return [rasterio.open(path, 'r').read(1) for path in files_paths]


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')

    # Run generation
    main(args)
