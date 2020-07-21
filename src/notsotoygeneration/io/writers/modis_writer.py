import os
from .scene_writer import SceneWriter
from ..utils import convert_modis_coordinate_to_aws_path, convert_date_to_aws_path


class MODISSceneWriter(SceneWriter):
    """Extends SceneReader by handling MODIS data type and directory structure

    For example, directory would usually be structured as :

        root
        └── 18                          # MODIS horizontal tile
            └── 04                      # MODIS vertical tile
                └── 2017                # Year
                    └── 1               # Month
                        ├── 1           # Day
                        │   └── 0/
                        │  ...
                        └── 31
                            └── 0/

    where each subdirectory has substructure :

        0/
        ├── scene.TIF       # Scene file (one or more)
        └── infos.json     # Infos file

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
        """Writes name of date subdirectory given date

        Args:
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        date_directory = convert_date_to_aws_path(date)
        return date_directory

    def _format_filename(self, filename, modis_coordinate, date):
        """Writes filename correponding to specified coordinates and date

        Args:
            filename (str): name of file to write in
            modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str

        """
        if not filename:
            filename = self._get_default_filename(modis_coordinate, date)
        return filename

    def _get_default_filename(self, modis_coordinate, date):
        """Composes default filename for writing file as concatenation of modis
        coordinate and date with file extension

        Args:
            modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        str_modis_coordinate = list(map(str, modis_coordinate))
        filename = '_'.join(str_modis_coordinate + [date])
        filename = filename + '.' + self.extension
        return filename

    def get_path_to_scene(self, modis_coordinate, date, filename):
        """Writes full path to information file corresponding to specified modis
        coordinate, date and band

        Args:
            modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd
            filename (str): name of file to write in

        Returns:
            type: str
        """
        modis_region_directory = self._format_region_directory(modis_coordinate)
        date_directory = self._format_date_directory(date)
        filename = self._format_filename(filename, modis_coordinate, date)
        path_to_scene = os.path.join(self.root, modis_region_directory, date_directory, filename)
        return path_to_scene

    def get_path_to_infos(self, modis_coordinate, date):
        """Writes full path to information file corresponding to specified modis
        coordinate and date

        Args:
            modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        modis_region_directory = self._format_region_directory(modis_coordinate)
        date_directory = self._format_date_directory(date)
        path_to_infos = os.path.join(self.root, modis_region_directory, date_directory, self._infos_filename)
        return path_to_infos
