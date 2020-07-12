from abc import ABC, abstractmethod
import rasterio


class SceneReader(ABC):
    """General class to access and read scenes. Scenes can be stored as jp2 or
    tif files. Root directory only is provided, further directory structure is
    left to be precised in child classes

    Implements an :

        - `open` method : loads and return raster correspondong to specified
            arguments

        - `with`-statement-like formalism to be used as for example :

            with scene_reader(scene_coordinate=coordinate, scene_date=date) as raster:
                    # manipulate your raster

            while `open` method is compatible with any type of argument, `with`
            statement only handles keyed arguments

    Args:
        root (str): root directory where scenes are stored
        extension (str): scenes files extensions {'jp2', 'tif', 'JP2', 'TIF'}
    """

    _scenes_extensions = {'jp2', 'tif'}

    def __init__(self, root, extension):
        self.root = root
        self.extension = extension

    @abstractmethod
    def get_path_to_scene(self, *args, **kwargs):
        """Returns path to scene corresponding to specified arguments
        """
        pass

    @abstractmethod
    def get_path_to_infos(self, *args, **kwargs):
        """Return path to information file corresponding to specified arguments
        """
        pass

    def open(self, *args, **kwargs):
        """Loads raster at path corresponding to specified arguments
        """
        file_path = self.get_path_to_scene(*args, **kwargs)
        raster = rasterio.open(file_path)
        return raster

    def __call__(self, **kwargs):
        """Allows to set keyed arguments as temporary private attributes which are
        then used to load raster from file with syntax :
            with self(**kwargs):
                # do your thing
        """
        self._tmp_access_kwargs = kwargs.copy()
        return self

    def __enter__(self):
        """Access raster and return it
        """
        self._raster = self.open(**self._tmp_access_kwargs)
        return self._raster

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown raster and delete temporary access attributes
        """
        self._raster.close()
        del self.__dict__['_tmp_access_kwargs']
        del self.__dict__['_raster']

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

    @property
    def extension(self):
        return self._extension

    @extension.setter
    def extension(self, extension):
        assert extension.lower() in self._scenes_extensions, f"Provided extension {extension} while only {self._scenes_extensions} are allowed"
        self._extension = extension
