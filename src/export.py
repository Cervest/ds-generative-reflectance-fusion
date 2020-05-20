import os
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from src.utils import mkdir, save_json, load_json


class ProductExport:
    """Handler for product image dumping during generation or derivation step

    Sets up an output directory structured as :
    ```
    directory_name/
    ├── frames/
    ├── annotations/
    └── index.json
    ```
    where:
        - `frames/`: generated imagery frames by time step
        - `annotations/`: mask annotation of each frame
        - `index.json`: mapping to frames respective annotation path by time
        step + descriptive characteristics of generated imagery product

    Args:
        output_dir (str): output directory
        astype (str): export type in {'h5', 'jpg'}
    """
    _frame_dirname = 'frames/'
    _annotation_dirname = 'annotations/'
    _frame_name = 'frame_{i:03d}'
    _annotation_name = 'annotation_{i:03d}.h5'
    _index_name = 'index.json'
    __frames_export_types__ = {'h5', 'jpg'}

    def __init__(self, output_dir, astype):
        if astype not in self.__frames_export_types__:
            raise TypeError("Unknown dumping type")
        self._output_dir = output_dir
        self._astype = astype

    def __enter__(self):
        self._setup_output_dir()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        return True

    def _setup_output_dir(self, output_dir=None, overwrite=False):
        """Builds output directory hierarchy structured as :

            directory_name/
            ├── frames/
            ├── annotations/
            └── index.json

        Args:
            output_dir (str): path to output directory
            overwrite (bool): if True and directory already exists, erases
                everything and recreates from scratch
        """
        output_dir = output_dir or self.output_dir
        frames_dir = os.path.join(output_dir, self._frame_dirname)
        annotations_dir = os.path.join(output_dir, self._annotation_dirname)
        mkdir(output_dir, overwrite=overwrite)
        mkdir(frames_dir)
        mkdir(annotations_dir)

    def _init_generation_index(self, product):
        """Initializes generation index as described above (this is the
        to be `index.json`)

        Returns:
            type: dict
        """
        index = {'features': {'width': product.size[0],
                              'height': product.size[1],
                              'nbands': product.nbands,
                              'horizon': product.horizon,
                              'ndigit': len(product),
                              'nframes': 0},
                 'files': dict()}
        self._index = index

    def add_to_index(self, idx, frame_name, annotation_name):
        """Records files paths into generation index to create unique mapping
        of frames and corresponding annotations by time step

        Args:
            idx (int): key mapping to frame and respective annotation paths
            frame_name (str)
            annotation_name (str)
        """
        frame_path = os.path.join(self._frame_dirname, frame_name)
        annotation_path = os.path.join(self._annotation_dirname, annotation_name)

        self._index['files'][idx] = {'frame': frame_path,
                                     'annotation': annotation_path}
        self._index['features']['nframes'] += 1

    def dump_array(self, array, dump_path):
        """Dumps numpy array following hdf5 protocol

        Args:
            array (np.ndarray)
            dump_path (str)
            name (str): Optional dataset name
        """
        with h5py.File(dump_path, 'w') as f:
            f.create_dataset('data', data=array)

    def dump_jpg(self, array, dump_path):
        """Dumps numpy array as jpg image
        Only compatible with 3-bands product

        Args:
            array (np.ndarray)
            dump_path (str)
        """
        if array.ndim != 3 or array.shape[-1] != 3:
            raise RuntimeError("RGB image generation only available for 3-bands products")
        img = Image.fromarray((array.clip(0, 1) * 255).astype(np.uint8), mode='RGB')
        img.save(dump_path)

    def dump_frame(self, frame, filename, astype=None):
        """Dumps numpy array of imagery frame at specified location
        Handles jpg format for 3-bands products only

        Args:
            frame (np.ndarray): array to dump
            filename (str): dumped file name
            astype (str): in {'h5', 'jpg'}
        """
        astype = astype or self.astype
        dump_path = os.path.join(self.output_dir, self._frame_dirname, filename)
        if astype == 'h5':
            self.dump_array(array=frame, dump_path=dump_path)
        elif astype == 'jpg':
            self.dump_jpg(array=frame, dump_path=dump_path)
        else:
            raise TypeError("Unknown dumping type")

    def dump_annotation(self, annotation, filename):
        """Dumps annotation mask under annotation directory

        Args:
            annotation (np.ndarray): array to dump
            filename (str): dumped file name

        Returns:
            type: Description of returned object.
        """
        dump_path = os.path.join(self.output_dir, self._annotation_dirname, filename)
        self.dump_array(array=annotation, dump_path=dump_path)

    def dump_index(self, index=None):
        """Simply saves index as json file under export directory

        Args:
            index (dict): dictionnary to dump as json (default: self.index)
        """
        if hasattr(self, '_index') and index is None:
            index = self._index
        index_path = os.path.join(self.output_dir, self._index_name)
        save_json(path=index_path, jsonFile=index)

    def set_index(self, index):
        self._index = index

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def astype(self):
        return self._astype


class ProductDataset(Dataset):
    """Dataset loading class for generated products

    Very straigthforward implementation to be adapted to product dumping
        format

    Args:
        root (str): path to directory where product has been dumped
    """
    def __init__(self, root):
        self._root = root
        index_path = os.path.join(root, ProductExport._index_name)
        self._index = load_json(index_path)
        self._frames_path = self._get_paths(file_type='frame')
        self._annotations_path = self._get_paths(file_type='annotation')

    def __getitem__(self, idx):
        """Loads frame and annotation arrays
        If doesn't exist, None value returned instead

        Args:
            idx (int): dataset index - corresponds to time step

        Returns:
            type: tuple[np.ndarray]
        """
        frame_path = self._frames_path[idx]
        annotation_path = self._annotations_path[idx]
        frame = self._load_array(path=frame_path)
        annotation = self._load_array(path=annotation_path)
        return frame, annotation

    def _load_array(self, path):
        """h5py loading protocol, if null path returns None

        Args:
            path (str): path to array to load

        Returns:
            type: np.ndarray
        """
        if path:
            with h5py.File(path, 'r') as f:
                array = f['data'][:]
        else:
            array = None
        return array

    def _get_paths(self, file_type):
        """Computes paths to frames and annotations. If non existing, fills
        with None

        Args:
            file_type (str): in {'frame', 'annotation'}

        Returns:
            type: str
        """
        path = dict()
        for key, file in self.index['files'].items():
            if file is None:
                filepath = None
            else:
                filepath = os.path.join(self.root, file[file_type])
            path.update({int(key): filepath})
        return path

    def __len__(self):
        return len(self._frames_path)

    @property
    def root(self):
        return self._root

    @property
    def index(self):
        return self._index
