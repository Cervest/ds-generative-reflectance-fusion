import os
from .scene_writer import SceneWriter
from ..format import AWSFormatter
from ..utils import convert_mgrs_coordinate_to_aws_path


class S2SceneWriter(AWSFormatter, SceneWriter):
    """Extends SceneWriter by handling Sentinel-2 data type and writing directory
    structure. We choose to follow AWS directory structure.
    For example, for MGRS coordinate '31UDQ' writing directory would be structured as :

        31                              # MGRS longitude
        └── U                           # MGRS latitude
            └── DQ                      # MGRS subgrid square identifier
                └── 2017                # Year
                    ├── 1               # Month
                    │   ├── 16          # Day
                    │   │   └── 0/      # Snapshot id (usually only one)
                    │   └── 6
                    │       └── 0/
                    └── 2
                        ├── 12
                        │   └── 0/
                        └── 22
                            └── 0/

    Where each snapshot id subdirectory contains:

        └── 0/
            ├── scene.jp2           # Scene files (one or more)
            └── productInfo.json    # Infos file

    Sentinel-2 bands information : https://www.satimagingcorp.com/satellite-sensors/other-satellite-sensors/sentinel-2a/

    Args:
        root (str): root directory where scenes are stored
    """

    _infos_filename = 'tileInfo.json'

    def __init__(self, root, extension='jp2'):
        super().__init__(root=root, extension=extension)

    def _format_location_directory(self, coordinate):
        """Writes name of mgrs subdirectory given mgrs coordinates

        Args:
            coordinate (str): MGRS coordinate formatted as '31TBF'

        Returns:
            type: str
        """
        location_directory = convert_mgrs_coordinate_to_aws_path(coordinate)
        return location_directory

    def _get_default_filename(self, mgrs_coordinate, date):
        """Composes default filename for writing file as concatenation of mgrs
        coordinate and date with file extension

        Args:
            coordinate (tuple[int]): modis coordinate as (horizontal tile, vertical tile)
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        filename = '_'.join([mgrs_coordinate, date, '0'])
        filename = filename + '.' + self.extension
        return filename

    def get_path_to_infos(self, coordinate, date):
        """Writes full path to information file corresponding to specified mgrs
        coordinate and date

        Args:
            coordinate (str): MGRS coordinate formatted as '31TBF'
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        mgrs_directory = self._format_location_directory(coordinate)
        date_directory = self._format_date_directory(date)
        path_to_infos = os.path.join(self.root, mgrs_directory, date_directory, self._infos_filename)
        return path_to_infos
