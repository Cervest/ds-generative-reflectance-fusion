import os
import h5py
from src.utils import load_json, save_json


class PatchExport:
    """Handler for imagery patch dumping during landsat and modis joined patch
    extraction phase

    Sets up an output directories structured as :
    ```
    output_dir/
    └── patch_directory/
        ├── modis/
        ├── landsat/
        └── index.json
    ```
    where:
        - `modis/`: patches time serie extracted from modis scenes for given patch location
        - `landsat/`: patches time serie extracted from modis scenes for given patch location
        - `index.json`: mapping to patches respective path for each time step

    Args:
        output_dir (str): output directory
    """

    _modis_dirname = 'modis'
    _landsat_dirname = 'landsat'
    _patch_dirname = 'patch_{idx:03d}'
    _index_name = 'index.json'

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def _format_patch_directory_path(self, patch_idx):
        """Writes full path to patch directory given patch index

        Args:
            patch_idx (int)

        Returns:
            type: str
        """
        patch_directory = self._patch_dirname.format(idx=patch_idx)
        patch_directory_path = os.path.join(self.output_dir, patch_directory)
        return patch_directory_path

    def setup_output_dir(self, patch_idx):
        """Builds output directory hierarchy structured as :

            patch_directory/
            ├── modis/
            ├── landsat/
            └── index.json

        Args:
            patch_idx (int)
        """
        patch_directory_path = self._format_patch_directory_path(patch_idx)
        modis_dir = os.path.join(patch_directory_path, self._modis_dirname)
        landsat_dir = os.path.join(patch_directory_path, self._landsat_dirname)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(modis_dir, exist_ok=True)
        os.makedirs(landsat_dir, exist_ok=True)

    def setup_index(self, patch_idx, patch_bounds):
        """Initializes generation index as described above (this is the
        to be `index.json`)

        If already exists, loads it instead

        Returns:
            type: dict
        """
        index_path = self._get_index_path(patch_idx)

        if os.path.exists(index_path):
            index = load_json(path=index_path)
        else:
            index = {'features': {'patch_idx': patch_idx,
                                  'patch_bounds': patch_bounds,
                                  'horizon': 0},
                     'files': dict()}
        return index

    def update_index(self, index, patch_idx, date):
        """Records files paths into generation index to create unique mapping
        of frames and corresponding annotations by time step

        Args:
            idx (int): key mapping to frame
            modis_name (str)
            landsat_name (str)
            target_name (str)
        """
        # Write realtive paths to frames
        filename = date + '.h5'
        modis_path = os.path.join(self._modis_dirname, filename)
        landsat_path = os.path.join(self._landsat_dirname, filename)
        target_path = os.path.join(self._landsat_dirname, filename)

        n_files = len(index['files'])
        index['files'][1 + n_files] = {'date': date,
                                       'modis': modis_path,
                                       'landsat': landsat_path,
                                       'target': target_path}
        index['features']['horizon'] = len(index['files'])
        return index

    def dump_array(self, array, dump_path):
        """Dumps numpy array following hdf5 protocol

        Args:
            array (np.ndarray)
            dump_path (str)
        """
        with h5py.File(dump_path, 'w') as f:
            f.create_dataset('data', data=array)

    def dump_patches(self, patch_idx, modis_patch, landsat_patch, date):
        """Dumps modis and landsat frames under patch directory and dated file naming

        Args:
            patch_idx (int)
            modis_patch (np.ndarray)
            landsat_patch (np.ndarray)
            date (str): date formatted as yyyy-mm-dd
        """
        filename = date + '.h5'
        patch_directory_path = self._format_patch_directory_path(patch_idx)
        modis_dump_path = os.path.join(patch_directory_path, self._modis_dirname, filename)
        landsat_dump_path = os.path.join(patch_directory_path, self._landsat_dirname, filename)
        self.dump_array(modis_patch, modis_dump_path)
        self.dump_array(landsat_patch, landsat_dump_path)

    def _get_index_path(self, patch_idx):
        """Writes path to index file

        Args:
            patch_idx (int)

        Returns:
            type: str
        """
        patch_directory_path = self._format_patch_directory_path(patch_idx)
        index_path = os.path.join(patch_directory_path, self._index_name)
        return index_path

    def dump_index(self, index, patch_idx):
        """Simply saves index as json file under export directory

        Args:
            index (dict): dictionnary to dump as json
        """
        index_path = self._get_index_path(patch_idx)
        save_json(path=index_path, jsonFile=index)

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir):
        self._output_dir = output_dir
