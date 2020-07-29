import os
import datetime
from .scene_reader import SceneReader
from ..format import ScenePathFormatter, AWSFormatter


class S1BandReader(ScenePathFormatter, SceneReader):
    """Sentinel-1 reading is a bit different as we currently only have access to
    a single location for a single year so this implementation stands aside
    from location/date/file logic. To be refined if needed in the future.

    Directory would be structured as :

        root
        ├── ascending_VH
        │   ├── S1A_Sigma0_VV_04Jan2018_descending.tif
        │   ├── ...
        │   └── S1A_Sigma0_VV_24Dec2018_descending.tif
        ├── ascending_VV
        │   ├── S1A_Sigma0_VV_04Jan2018_descending.tif
        │   ├── ...
        │   └── S1A_Sigma0_VV_24Dec2018_descending.tif
        ├── descending_VH
        │   ├── S1A_Sigma0_VH_02Jan2018_descending.tif
        │   ├── ...
        │   └── S1A_Sigma0_VH_28Dec2018_descending.tif
        └── descending_VV
            ├── S1A_Sigma0_VV_02Jan2018_descending.tif
            ├── ...
            └── S1A_Sigma0_VV_28Dec2018_descending.tif

    where {'ascending', 'descending'} denotes the orbit on of the device when
    shot was taken and {'VH', 'VV'} its polarization.

    Each subdirectory can be used as if it was a different band from the same snapshot

    Args:
        root (str): root directory where scenes are stored
    """

    _filename_root = 'S1A_Sigma0'

    def __init__(self, root, extension='tif'):
        super().__init__(root=root, extension=extension)

    def _format_location_directory(self, location):
        raise NotImplementedError

    def _format_date_directory(self, date):
        raise NotImplementedError

    def _format_date(self, date):
        """Formats date from '2018-01-02' to '02Jan2018' to write files names

        Args:
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        year, month, day = list(map(int, date.split('-')))
        date = datetime.datetime(year, month, day)
        return date.strftime("%d%b%Y")

    def _format_band_directory(self, orbit, polarization):
        """Writes directory name corresponding to specified orbit and polarization

        Args:
            orbit (str): {'ascending', 'descending'}
            polarization (str): {'VH', 'VV'}

        Returns:
            type: str
        """
        return '_'.join([orbit, polarization])

    def _get_default_filename(self, orbit, polarization, date):
        """Writes name of file in subdirectory corresponding to specified orbit,
        polarization and date

        Args:
            orbit (str): {'ascending', 'descending'}
            polarization (str): {'VH', 'VV'}
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str

        """
        date = self._format_date(date)
        filename = '_'.join([self._filename_root, polarization, date, orbit])
        filename = filename + '.' + self.extension
        return filename

    def get_path_to_scene(self, coordinate, date, filename, orbit, polarization):
        """Writes full path to scene corresponding to specified orbit,
        polarization and date

        Args:
            date (str): date formatted as yyyy-mm-dd
            orbit (str): {'ascending', 'descending'}
            polarization (str): {'VH', 'VV'}

        Returns:
            type: str

        """
        band_directory = self._format_band_directory(orbit, polarization)
        filename = self._format_filename(filename=None,
                                         orbit=orbit,
                                         polarization=polarization,
                                         date=date)
        path_to_scene = os.path.join(self.root, band_directory, filename)
        return path_to_scene

    def get_path_to_infos(self, orbit, polarization, date):
        raise NotImplementedError("No information provided on Sentinel-1 scenes")


class S1SceneReader(AWSFormatter, SceneReader):
    """Extends SceneReader by handling S1 data type and directory structure

    For example, directory would usually be structured as :

        root
        └── 2018            # Year
            └── 01          # Month
                └── 01      # Day
                    └── 0/  # Snapshot id

    where each subdirectory has substructure :

        0/
        └── 2018-01-01_0.tif

    Args:
        root (str): root directory where scenes are stored
    """
    def __init__(self, root, extension='tif'):
        super().__init__(root=root, extension=extension)

    def _format_location_directory(self, coordinate):
        """Returns empty location directory

        Returns:
            type: str
        """
        return str()

    def _get_default_filename(self, date):
        """Composes default filename for writing file as concatenation of date
        and file extension

        Args:
            dates (list[str]): list of dates formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        filename = date + '.' + self.extension
        return filename
