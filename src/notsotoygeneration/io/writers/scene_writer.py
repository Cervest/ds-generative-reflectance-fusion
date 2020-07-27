import os
import rasterio


class SceneWriter:
    """General class to access and write scenes

    Implements an :

        - `open` method : intializes and returns writing mode raster corresponding
            to specified arguments

        - `with`-statement-like formalism to be used as for example :

            with scene_writer(meta=meta, scene_coordinate=coordinate, scene_date=date) as raster:
                # manipulate your raster

            while `open` method is compatible with any type of argument, `with`
            statement only handles keyed arguments

    Both methods must be provided with a raster metadata dictionnary which would
    be typically formatted as :

        {'driver': 'JP2OpenJPEG',                   # Writing_driver_type, e.g. {'JP2OpenJPEG', 'GTiff'}
         'dtype': 'uint16',
         'width': 10980,
         'height': 10980,
         'count': 1,                                # Number of bands in raster
         'crs': CRS.from_epsg(32631),               # Coordinate reference system
         'transform': Affine(10.0, 0.0, 600000.0,   # Correction transform ? TBC
                0.0, -10.0, 5500020.0)}
    """
    def open(self, meta, *args, **kwargs):
        """Loads writing raster at path corresponding to specified arguments

        Args:
            meta (dict): writing raster metadata
        """
        file_path = self.get_path_to_scene(*args, **kwargs)
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        raster = rasterio.open(file_path, 'w', **self._meta)
        return raster

    def __call__(self, meta, **kwargs):
        """Allows to set keyed arguments as temporary private attributes which are
        then used to load raster from file with syntax :
            with self(meta, **kwargs):
                # do your thing
        """
        self._meta = meta
        self._tmp_access_kwargs = kwargs.copy()
        return self

    def __enter__(self):
        """Access raster and return it
        """
        self._raster = self.open(meta=self._meta,
                                 **self._tmp_access_kwargs)
        return self._raster

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown raster and delete temporary access attributes
        """
        self._raster.close()
        del self.__dict__['_tmp_access_kwargs']
        del self.__dict__['_meta']
        del self.__dict__['_raster']
