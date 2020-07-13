import os
from .scene_writer import SceneWriter
from .utils import convert_mgrs_coordinate_to_aws_path, convert_date_to_aws_path


class S2SceneWriter(SceneWriter):
    """Extends SceneWriter by handling Sentinel-2 data type and directory structure
    We choose to follow AWS directory structure.
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

    def _format_mgrs_directory(self, mgrs_coordinate):
        """Writes name of mgrs subdirectory given mgrs coordinates

        Args:
            mgrs_coordinate (str): coordinate formatted as '31TBF'

        Returns:
            type: str
        """
        mgrs_directory = convert_mgrs_coordinate_to_aws_path(mgrs_coordinate)
        return mgrs_directory

    def _format_date_directory(self, date):
        """Writes name of date subdirectory given date

        Args:
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        date_directory = convert_date_to_aws_path(date)
        return date_directory

    def _format_filename(self, filename, mgrs_coordinate, date):
        """Makes sure name of file to write is properly formatted

        Args:
            filename (str): name of file to write in

        Returns:
            type: str
        """
        if not filename:
            filename = self._get_default_filename(mgrs_coordinate, date)
        return filename

    def _get_default_filename(self, mgrs_coordinate, date):
        """Composes default filename for writing file as concatenation of mgrs
        coordinate and date with file extension

        Args:
            mgrs_coordinate (str): coordinate formatted as '31TBF'
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        filename = '_'.join([mgrs_coordinate, date, '0'])
        filename = filename + '.' + self.extension
        return filename

    def get_path_to_scene(self, mgrs_coordinate, date, filename):
        """Writes full path to writing scene corresponding to specified mgrs
        coordinate, date and filename

        Args:
            mgrs_coordinate (str): coordinate formatted as '31TBF'
            date (str): date formatted as yyyy-mm-dd
            filename (str): name of file to write in

        Returns:
            type: str
        """
        mgrs_directory = self._format_mgrs_directory(mgrs_coordinate)
        date_directory = self._format_date_directory(date)
        filename = self._format_filename(filename)
        path_to_scene = os.path.join(self.root, mgrs_directory, date_directory, filename)
        return path_to_scene

    def get_path_to_infos(self, mgrs_coordinate, date):
        """Writes full path to information file corresponding to specified mgrs
        coordinate and date

        Args:
            mgrs_coordinate (str): coordinate formatted as '31TBF'
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        mgrs_directory = self._format_mgrs_directory(mgrs_coordinate)
        date_directory = self._format_date_directory(date)
        path_to_infos = os.path.join(self.root, mgrs_directory, date_directory, self._infos_filename)
        return path_to_infos
