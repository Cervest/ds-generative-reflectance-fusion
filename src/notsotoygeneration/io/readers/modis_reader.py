import os
import datetime
from .scene_reader import SceneReader
from ..utils import convert_modis_coordinate_to_aws_path
from src.utils import load_json


class MODISBandReader(SceneReader):
    """Extends SceneReader by handling MODIS data type and directory structure

    For example, directory would usually be structured as :

        root
        └── 18                  # MODIS horizontal tile
            └── 04              # MODIS vertical tile
                ├── 2018001     # Date directory as year + day_of_the_year
                ├── 2018002
                ├── ...
                └── 2018365

    where each subdirectory has substructure :

        2018001
        ├── MCD43A4.A2018001.h18v04.006.2018010031310_B01.TIF       # Band files
        ├── MCD43A4.A2018001.h18v04.006.2018010031310_B02.TIF
        ├── ...
        ├── MCD43A4.A2018001.h18v04.006.2018010031310_B07.TIF
        ├── MCD43A4.A2018001.h18v04.006.2018010031310_B07qa.TIF
        ├── MCD43A4.A2018001.h18v04.006.2018010031310_meta.json     # Infos file
        └── index.html

    MODIS Bands informations : https://modis.gsfc.nasa.gov/about/specifications.php

    Args:
        root (str): root directory where scenes are stored
    """
    def __init__(self, root, extension='TIF'):
        super().__init__(root=root, extension=extension)

    def _format_region_directory(self, modis_coordinate):
        """Write directory corresponding to coordinates

        Args:
            modis_coordinate (tuple[int]): (horizontal tile, vertical tile)

        Returns:
            type: str
        """
        modis_region_directory = convert_modis_coordinate_to_aws_path(modis_coordinate)
        return modis_region_directory

    def _format_date_directory(self, date):
        """Write date subdirectory name converting date in yyyydoy format
        where doy = day of year

        Args:
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        year, month, day = list(map(int, date.split('-')))
        date = datetime.datetime(year, month, day)
        day_of_year = date.timetuple().tm_yday
        date_directory = str(year) + '{0:03d}'.format(day_of_year)
        return date_directory

    def _format_directory(self, modis_coordinate, date):
        """Writes and joins names of coordinate subdirectory and date subdirectory

        Args:
            modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        region_directory = self._format_region_directory(modis_coordinate)
        date_directory = self._format_date_directory(date)
        directory = os.path.join(region_directory, date_directory)
        return directory

    def _format_filename(self, band, modis_coordinate, date):
        """Writes filename correponding to specified coordinates and date

        Args:
            band (str): name of band (e.g. 'B02')
            modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str

        """
        file_name_root = self._get_file_name_root(modis_coordinate, date)
        filename = file_name_root + '_' + band + '.' + self.extension
        return filename

    def _get_file_name_root(self, modis_coordinate, date):
        """Loads metadata to extact naming root of band files
        (e.g. 'MCD43A4.A2018001.h18v04.006.2018010031310' in docstring example)

        Args:
            modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        infos_path = self.get_path_to_infos(modis_coordinate, date)
        meta_data = load_json(infos_path)
        producer_granule_id = meta_data['producer_granule_id']
        file_name_root = '.'.join(producer_granule_id.split('.')[:-1])
        return file_name_root

    def get_path_to_infos(self, modis_coordinate, date):
        """Writes full path to information file corresponding to specified modis
        coordinate and date

        Args:
            modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        directory_name = self._format_directory(modis_coordinate, date)
        directory_path = os.path.join(self.root, directory_name)
        file_paths = os.listdir(directory_path)
        infos_filename = next(filter(lambda x: x.endswith('json'), file_paths))
        path_to_infos = os.path.join(directory_path, infos_filename)
        return path_to_infos

    def get_path_to_scene(self, modis_coordinate, date, band):
        """Writes full path to information file corresponding to specified modis
        coordinate, date and band

        Args:
            mgrs_coordinate (str): coordinate formatted as '31TBF'
            date (str): date formatted as yyyy-mm-dd
            band (str): name of band (e.g. 'B02')

        Returns:
            type: str
        """
        directory = self._format_directory(modis_coordinate, date)
        filename = self._format_filename(band, modis_coordinate, date)
        path_to_scene = os.path.join(self.root, directory, filename)
        return path_to_scene
