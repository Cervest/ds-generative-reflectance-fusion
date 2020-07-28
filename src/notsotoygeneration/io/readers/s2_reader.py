import os
from .scene_reader import SceneReader, BandReader
from ..format import ScenePathFormatter, AWSFormatter
from ..utils import convert_mgrs_coordinate_to_aws_path


class S2BandReader(ScenePathFormatter, BandReader):
    """Extends SceneReader by handling Sentinel-2 data type and directory structure

    For example, directory would usually be structured as :

    root
    └── 31UDQ                       # MGRS coordinate
        └── 2017-1-16_0             # Date
            ├── B01.jp2             # Band files
            ├── B02.jp2
            ├── B03.jp2
            ├── B04.jp2
            ├── ...
            ├── B11.jp2
            ├── B12.jp2
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
        return coordinate

    def _format_date_directory(self, date):
        """Writes name of date subdirectory given date

        Args:
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        if not date.endswith('_0'):
            date = date + '_0'
        return date

    def _format_filename(self, filename):
        """Makes sure name of file to load is properly formatted

        Args:
            filename (str): name of band to load (e.g. 'B02', 'B03', etc)

        Returns:
            type: str
        """
        if not filename.endswith(self.extension):
            filename = '.'.join([filename, self.extension])
        return filename

    def _get_default_filename(self, *args, **kwargs):
        raise NotImplementedError

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


class S2SceneReader(AWSFormatter, SceneReader):
    """Extends SceneReader by handling Sentinel-2 data type and directory structure

    For example, directory would usually be structured as :

        root
        └── 31                          # MGRS latitude
            └── U                       # MGRS longitude
                └── EQ                  # Subgrid id
                    └── 2018            # Year
                        └── 2           # Month
                            └── 22      # Day
                                └── 0/  # Snapshot id

    where each subdirectory has substructure :

        0/
        └── 31UEQ_2018-2-22_0.jp2

    Args:
        root (str): root directory where scenes are stored
    """

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

    def _get_default_filename(self, coordinate, date):
        """Composes default filename for writing file as concatenation of mgrs
        coordinate and date with file extension

        Args:
            coordinate (str): MGRS coordinate formatted as '31TBF'
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        filename = '_'.join([coordinate, date, '0'])
        filename = filename + '.' + self.extension
        return filename

    def get_path_to_infos(self, mgrs_coordinate, date):
        raise NotImplementedError("No infos sorry")
