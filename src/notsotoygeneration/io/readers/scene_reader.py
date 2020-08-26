from abc import ABC, abstractmethod
import rasterio
from rasterio.io import MemoryFile


class SceneReader(ABC):
    """General class to access and read imagery scenes

    Implements an :

        (1) `open` method : loads and return raster corresponding to specified
            arguments

        (2) `with`-like-statement formalism to be used as for example :

        ```
            with scene_reader(coordinate=coordinate, date=date) as raster:
                # manipulate your raster
        ```

        Note that while `open` method is compatible with any type of argument,
            `with` statement only supports keyed arguments
    """

    @abstractmethod
    def get_path_to_scene(self, coordinate, date, filename, *args, **kwargs):
        """Writes path to scene file as concatenation of location directory,
        date directory and filename

        Args:
            coordinate (object): location related object
            date (object): date related object
            filename (str): name of file to access

        Returns:
            type: str

        """
        pass

    @abstractmethod
    def get_path_to_infos(self, coordinate, date, *args, **kwargs):
        """Writes path to information file contained in location and date directory

        Args:
            coordinate (object): coordinate information - to be precised in child class
            date (object): date information - to be precised in child class

        Returns:
            type: str

        """
        pass

    def open(self, coordinate, date, filename=None, *args, **kwargs):
        """Loads raster at path corresponding to specified arguments

        Args:
            coordinate (object): coordinate information - to be precised in child class
            date (object): date information - to be precised in child class
            filename (str): name of file to access

        Returns:
            type: rasterio.io.DatasetReader
        """
        file_path = self.get_path_to_scene(coordinate, date, filename, *args, **kwargs)
        raster = rasterio.open(file_path)
        return raster

    def get_meta(self, coordinate, date, filename, *args, **kwargs):
        """Short utility to retrieve raster file metadata only

        Args:
            coordinate (object): coordinate information - to be precised in child class
            date (object): date information - to be precised in child class
            filename (str): name of file to access

        Returns:
            type: dict
        """
        with self(coordinate=coordinate, date=date, filename=filename, **kwargs) as raster:
            meta = raster.meta
        return meta

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


class BandReader(SceneReader):
    """Extends SceneReader by for specific usage of separate band files reading"""

    def _open_and_stack_bands(self, coordinate, date, bands):
        """Loads band file rasters at path specified by arguments and stacks
        bands into single raster

        Args:
            coordinate (object): coordinate information - to be precised in child class
            date (object): date information - to be precised in child class
            bands (list[str]): list of band files to load

        Returns:
            type: rasterio.io.DatasetReader
        """
        # Load metadata from first band - assume similar for other bands
        meta = self.get_meta(coordinate=coordinate, date=date, band=bands[0])
        meta.update({'count': len(bands)})

        # Create temporary in-memory file
        memory_file = MemoryFile()

        # Write new scene containing all bands from directory specified in kwargs
        with memory_file.open(**meta) as target_raster:
            for idx, band in enumerate(bands):
                with self(coordinate=coordinate, date=date, bands=[band]) as source_raster:
                    target_raster.write_band(idx + 1, source_raster.read(1))
        return memory_file.open()

    def get_meta(self, coordinate, date, band, *args, **kwargs):
        """Short utility to retrieve raster file metadata only
            Overrides parent method to support `band` argument
        """
        with self(coordinate=coordinate, date=date, bands=[band], **kwargs) as raster:
            meta = raster.meta
        return meta

    def open(self, coordinate, date, bands):
        """Loads raster at path corresponding to specified arguments
           Overrides parent method to support `bands` argument instead of `filename`

        Args:
            coordinate (object): coordinate information - to be precised in child class
            date (object): date information - to be precised in child class
            bands (list[str]): list of bands to load

        Returns:
            type: rasterio.io.DatasetReader
        """
        if len(bands) == 1:
            # If single band, load it as such
            raster = super().open(coordinate=coordinate, date=date, filename=bands[0])
        else:
            # Else stack them together into single raster
            raster = self._open_and_stack_bands(coordinate=coordinate, date=date, bands=bands)
        return raster
