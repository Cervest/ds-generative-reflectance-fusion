import os
import datetime
from .scene_reader import SceneReader
from ..utils import convert_date_to_aws_path


class S1BandReader(SceneReader):
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

    def _write_filename(self, orbit, polarization, date):
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
        file_name = '_'.join([self._filename_root, polarization, date, orbit])
        file_name = file_name + '.' + self.extension
        return file_name

    def get_path_to_scene(self, orbit, polarization, date):
        """Writes full path to scene corresponding to specified orbit,
        polarization and date

        Args:
            orbit (str): {'ascending', 'descending'}
            polarization (str): {'VH', 'VV'}
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str

        """
        band_directory = self._format_band_directory(orbit, polarization)
        filename = self._write_filename(orbit, polarization, date)
        path_to_scene = os.path.join(self.root, band_directory, filename)
        return path_to_scene

    def get_path_to_infos(self, orbit, polarization, date):
        raise NotImplementedError("No information provided on Sentinel-1 scenes")


class S1SceneReader(SceneReader):
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

    def _format_date_directory(self, date):
        """Write date subdirectory name converting date in yyyydoy format
        where doy = day of year

        Args:
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        date_directory = convert_date_to_aws_path(date)
        return date_directory

    def _format_filename(self, filename, date):
        """Makes sure name of file to write is properly formatted

        Args:
            filename (str): name of file to write in
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        if not filename:
            filename = self._get_default_filename(date)
        return filename

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

    def get_path_to_scene(self, date, filename=None):
        """Writes full path to scene corresponding to specified orbit,
        polarization and date

        Args:
            date (str): date formatted as yyyy-mm-dd
            filename (str): name of file to read from

        Returns:
            type: str
        """
        date_directory = self._format_date(date)
        filename = self._format_filename(filename, date)
        path_to_scene = os.path.join(self.root, date_directory, filename)
        return path_to_scene

    def get_path_to_infos(self, date):
        raise NotImplementedError("No information provided on Sentinel-1 scenes")
