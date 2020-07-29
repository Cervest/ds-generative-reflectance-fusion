from .scene_writer import SceneWriter
from ..format import AWSFormatter
from ..utils import convert_modis_coordinate_to_aws_path


class MODISSceneWriter(AWSFormatter, SceneWriter):
    """Extends SceneWriter by handling MODIS data type and directory structure

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
    def __init__(self, root, extension='tif'):
        super().__init__(root=root, extension=extension)

    def _format_location_directory(self, coordinate):
        """Write directory corresponding to coordinates

        Args:
            coordinate (tuple[int]): modis coordinate as (horizontal tile, vertical tile)

        Returns:
            type: str
        """
        modis_region_directory = convert_modis_coordinate_to_aws_path(coordinate)
        return modis_region_directory

    def _get_default_filename(self, coordinate, date):
        """Composes default filename for writing file as concatenation of modis
        coordinate and date with file extension

        Args:
            coordinate (tuple[int]): modis coordinate as (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        str_modis_coordinate = list(map(str, coordinate))
        filename = '_'.join(str_modis_coordinate + [date])
        filename = filename + '.' + self.extension
        return filename
