"""
Description : Iterates over paired patch arrays extracted from Landsat and MODIS data
    and splits them into individual raster by band in a structured directory

Usage: prepare_inputs.py --o=<output_directory> --patch_dir=<patches_directory> --estarfm_out=<output_directory_for_starfm>

Options:
  -h --help                                             Show help.
  --version                                             Show version.
  --shapefile=<raw_files_directory>                     Path to shape file
  --o=<output_directory>                                Output directory
  --patch_dir=<path_to_scenes_directory>                Directory where patches have been dumped at patch extraction step
  --estarfm_out=<output_directory_for_starfm>           Output directory for STARFM predicted frames
"""
import os
import sys
from docopt import docopt
import logging
import rasterio
from progress.bar import Bar

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../")
sys.path.append(base_dir)

from src.notsotoygeneration.preprocessing.patch_extraction.export import PatchDataset, PatchExport


def main(args):
    patches_directories = [os.path.join(args['--patch_dir'], x) for x in os.listdir(args['--patch_dir'])]
    bar = Bar(f"Splitting patches bands into rasters", max=len(patches_directories))
    for patch_dir in patches_directories:
        patch_dataset = PatchDataset(root=patch_dir)
        export = ESTARFMPatchExport(output_dir=args['--o'])
        export_patch_to_rasters_by_band(patch_dataset, export, args['--estarfm_out'])
        bar.next()


def export_patch_to_rasters_by_band(patch_dataset, export, starfm_output_dir):
    patch_idx = patch_dataset.index['features']['patch_idx']
    patch_bounds = patch_dataset.index['features']['patch_bounds']
    if patch_dataset.index['features']['horizon'] < 3:
        return
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
            index = export.update_index(index, patch_idx, date, band)

    # Dump index
    write_starfm_params(export, patch_idx, index, starfm_output_dir)
    export.dump_index(index, patch_idx)


def write_starfm_params(export, patch_idx, index, output_dir):
    patch_dir = export._format_patch_directory_path(patch_idx)
    files = index['files']
    dates = list(files.keys())

    params_dir = os.path.join(patch_dir, 'params')
    os.makedirs(params_dir, exist_ok=True)

    for i in range(len(dates) - 2):

        for band_idx in range(1, 5):
            text = "ESTARFM_PARAMETER_START\n\n"
            text += "NUM_IN_PAIRS = 2\n\n"

            # band_idx = str(band_idx)

            last_date = dates[i]
            modis_last = os.path.join(patch_dir, files[last_date]['modis'][band_idx])
            landsat_last = os.path.join(patch_dir, files[last_date]['landsat'][band_idx])

            next_date = dates[i + 2]
            modis_next = os.path.join(patch_dir, files[next_date]['modis'][band_idx])
            landsat_next = os.path.join(patch_dir, files[next_date]['landsat'][band_idx])

            pred_date = dates[i + 1]
            modis_pred = os.path.join(patch_dir, files[pred_date]['modis'][band_idx])
            landsat_pred = os.path.join(output_dir, f"patch_{patch_idx}", pred_date, f"B{band_idx}.tif")
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

    def update_index(self, index, patch_idx, date, band):
        """Records files paths into generation index to create unique mapping
        of frames and corresponding annotations by time step

        Args:
            idx (int): key mapping to frame
            modis_name (str)
            landsat_name (str)
            target_name (str)
        """
        # Write realtive paths to frames
        filename = band + '.tif'
        modis_path = os.path.join(self._modis_dirname, date, filename)
        landsat_path = os.path.join(self._landsat_dirname, date, filename)

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

    def dump_patches(self, patch_idx, modis_band, landsat_band, date, band):
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
