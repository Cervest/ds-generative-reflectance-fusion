"""
Description : Prepares patches to ESTARFM inputs format by:
    (1) Loading each individual patch array from h5py file
    (2) Splitting them by band and saving bands separately as .tif rasters
    (3) Writing ESTARFM execution parameters file for each patch

Usage: prepare_inputs.py --patch_dir=<source_patches_directory> --o=<output_directory> --estarfm_out=<output_directory_for_estarfm_execution>

Options:
  --patch_dir=<path_to_scenes_directory>                Directory where patches have been dumped at patch extraction step
  --o=<output_directory>                                Output directory
  --estarfm_out=<output_directory_for_starfm>           Output directory for ESTARFM predicted images
"""
import os
import sys
from docopt import docopt
import logging
import rasterio
from progress.bar import Bar

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../")
sys.path.append(base_dir)

from src.prepare_data.preprocessing.patch_extraction import PatchDataset, PatchExport


def main(args):
    # Get list of path to patches directories
    patches_directories = [os.path.join(args['--patch_dir'], x) for x in os.listdir(args['--patch_dir'])]
    bar = Bar(f"Preparing patches for ESTARFM", max=len(patches_directories))

    # Setup export utility
    export = ESTARFMPatchExport(output_dir=args['--o'])

    for patch_dir in patches_directories:
        patch_dataset = PatchDataset(root=patch_dir)
        export_patch_to_rasters_by_band(patch_dataset, export, args['--estarfm_out'])
        bar.next()


def export_patch_to_rasters_by_band(patch_dataset, export, estarfm_output_dir):
    """Loads each patch from patch_dataset, splits them by bands and saves bands
        separately into rasters along with ESTARFM input parameter file

    Args:
        patch_dataset (PatchDataset)
        export (ESTARFMPatchExport)
        estarfm_output_dir (str): output directory for ESTARFM execution

    """
    patch_idx = patch_dataset.index['features']['patch_idx']
    patch_bounds = patch_dataset.index['features']['patch_bounds']

    if patch_dataset.index['features']['horizon'] < 3:
        # Discard datasets with less than 3 time steps
        return

    # Initialize output directory and output index
    export.setup_output_dir(patch_idx)
    index = export.setup_index(patch_idx=patch_idx, patch_bounds=patch_bounds)

    for idx, files_infos in patch_dataset.index['files'].items():
        # Get patches arrays
        date = files_infos['date']
        modis_patch, landsat_patch = patch_dataset[int(idx) - 1]

        # Bands first
        modis_patch = modis_patch.transpose(2, 0, 1)
        landsat_patch = landsat_patch.transpose(2, 0, 1)

        # Dump each band separately
        for band_idx, (modis_band, landsat_band) in enumerate(zip(modis_patch, landsat_patch)):
            band = 'B' + str(band_idx + 1)
            export.dump_patches(patch_idx, modis_band, landsat_band, date, band)
            index = export.update_index(index, date, band)

    # Write ESTARFM input parameters and dump index
    write_estarfm_params(export, patch_idx, index, estarfm_output_dir)
    export.dump_index(index, patch_idx)


def write_estarfm_params(export, patch_idx, index, estarfm_output_dir):
    """Writes and dumps ESTARFM input parameter file defining execution
    See here for parameters files format : https://github.com/HPSCIL/cuESTARFM

    Args:
        export (ESTARFMPatchExport)
        patch_idx (int): index of patch considered
        index (dict): patch directory information index
        estarfm_output_dir (str): output directory for ESTARFM execution

    """
    # Get path to patch directory and list of dates
    patch_dir = export._format_patch_directory_path(patch_idx)
    files = index['files']
    dates = list(files.keys())

    # Initialize separate directory for parameter files
    params_dir = os.path.join(patch_dir, 'params')
    os.makedirs(params_dir, exist_ok=True)

    # ESTARFM requires 2 input pairs to first and last dates are discarded
    for i in range(len(dates) - 2):
        for band_idx in range(1, 5):
            text = "ESTARFM_PARAMETER_START\n\n"
            text += "NUM_IN_PAIRS = 2\n\n"

            last_date = dates[i]
            modis_last = os.path.join(patch_dir, files[last_date]['modis'][band_idx])
            landsat_last = os.path.join(patch_dir, files[last_date]['landsat'][band_idx])

            next_date = dates[i + 2]
            modis_next = os.path.join(patch_dir, files[next_date]['modis'][band_idx])
            landsat_next = os.path.join(patch_dir, files[next_date]['landsat'][band_idx])

            pred_date = dates[i + 1]
            modis_pred = os.path.join(patch_dir, files[pred_date]['modis'][band_idx])
            landsat_pred = os.path.join(estarfm_output_dir, 'patch_{idx:03d}'.format(idx=patch_idx), pred_date, f"B{band_idx}.tif")
            os.makedirs(os.path.dirname(landsat_pred), exist_ok=True)

            text += f"IN_PAIR_MODIS_FNAME = {modis_last} {modis_next}\n\n"
            text += f"IN_PAIR_LANDSAT_FNAME = {landsat_last} {landsat_next}\n\n"

            text += f"IN_PDAY_MODIS_FNAME = {modis_pred}\n\n"

            text += f"OUT_PDAY_LANDSAT_FNAME = {landsat_pred}\n\n"

            text += "The_width_of_searching_window = 51\n\n"
            text += "Assumed_number_of_classifications = 6\n\n"
            text += "sensor_uncertain = 0.0028\n\n"
            text += "NODATA = 0\n\n"
            text += "G_Type = GTIff\n\n"

            text += "ESTARFM_PARAMETER_END"

            dump_path = os.path.join(params_dir, f"params_{pred_date}_B{band_idx}.txt")
            with open(dump_path, 'w') as f:
                f.write(text)


class ESTARFMPatchExport(PatchExport):
    """Extends PatchExport by handling ESTARFM input format
    """

    def update_index(self, index, date, band):
        """Updates patch directory information index when saving new band file

        Args:
            index (dict): patch directory information index
            date (str): date of file to record formatted as yyyy-mm-dd
            band (str): band corresponding to file to record
        """
        # Write realtive paths to frames
        filename = band + '.tif'
        modis_path = os.path.join(self._modis_dirname, date, filename)
        landsat_path = os.path.join(self._landsat_dirname, date, filename)

        # If band files with same date have already been recorded
        if date in index['files']:
            n_files = len(index['files'][date]['modis'])
            index['files'][date]['modis'].update({1 + n_files: modis_path})
            index['files'][date]['landsat'].update({1 + n_files: landsat_path})

        else:
            index['files'][date] = {}
            index['files'][date]['modis'] = {1: modis_path}
            index['files'][date]['landsat'] = {1: landsat_path}
        index['features']['horizon'] = len(index['files'][date]['modis'])
        return index

    def dump_patches(self, patch_idx, landsat_band, modis_band, date, band):
        """Dumps co-registered Landsat and MODIS band files as single band
            rasters within patch subdirectory corresponding to specified date
            and band

        Args:
            patch_idx (int): index of patch considered
            landsat_band (np.ndarray): (height, width)
            modis_band (np.ndarray): (height, width)
            date (str): date of file to record formatted as yyyy-mm-dd
            band (str): band corresponding to files to dump
        """
        # Make metadata
        meta = {'driver': 'ENVI',
                'dtype': 'int16',
                'width': modis_band.shape[0],
                'height': modis_band.shape[1],
                'count': 1}

        # Write files paths
        filename = band + '.tif'
        patch_directory_path = self._format_patch_directory_path(patch_idx)

        modis_dump_dir = os.path.join(patch_directory_path, self._modis_dirname, date)
        landsat_dump_dir = os.path.join(patch_directory_path, self._landsat_dirname, date)

        os.makedirs(modis_dump_dir, exist_ok=True)
        os.makedirs(landsat_dump_dir, exist_ok=True)

        modis_dump_path = os.path.join(modis_dump_dir, filename)
        landsat_dump_path = os.path.join(landsat_dump_dir, filename)

        # Dump as single band rasters
        with rasterio.open(modis_dump_path, 'w', **meta) as raster:
            raster.write_band(1, modis_band)
        with rasterio.open(landsat_dump_path, 'w', **meta) as raster:
            raster.write_band(1, landsat_band)


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')

    # Run generation
    main(args)
