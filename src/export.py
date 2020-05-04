import os
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from src.utils import mkdir, save_json, load_json


class ProductExport:
    """Handler for product image dumping during generation or derivation step

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
        """Initializes generation index

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
        """Records file names into index

        Args:
            idx (int)
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
        img = Image.fromarray((array * 255).astype(np.uint8), mode='RGB')
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

    def dump_index(self):
        """Simply saves index as json file under export directory

        Args:
            index (dict)
        """
        index_path = os.path.join(self.output_dir, self._index_name)
        save_json(path=index_path, jsonFile=self._index)

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
        self._frames_path = {int(key): os.path.join(root, file['frame'])
                             for (key, file) in self.index['files'].items()}
        self._annotations_path = {int(key): os.path.join(root, file['annotation'])
                                  for (key, file) in self.index['files'].items()}

    def __getitem__(self, idx):
        frame_path = self._frames_path[idx]
        annotation_path = self._annotations_path[idx]
        with h5py.File(frame_path, 'r') as f:
            frame = f['data'][:]
        with h5py.File(annotation_path, 'r') as f:
            annotation = f['data'][:]
        return frame, annotation

    def __len__(self):
        return len(self._files_path)

    @property
    def root(self):
        return self._root

    @property
    def index(self):
        return self._index