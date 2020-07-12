import os
import datetime
from .scene_reader import SceneReader


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

    where {'ascending', 'descending'} denotes the motion on of the device when
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

    def _format_band_directory(self, motion, polarization):
        """Writes directory name corresponding to specified motion and polarization

        Args:
            motion (str): {'ascending', 'descending'}
            polarization (str): {'VH', 'VV'}

        Returns:
            type: str
        """
        return '_'.join([motion, polarization])

    def _write_filename(self, motion, polarization, date):
        """Writes name of file in subdirectory corresponding to specified motion,
        polarization and date

        Args:
            motion (type): {'ascending', 'descending'}
            polarization (str): {'VH', 'VV'}
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str

        """
        date = self._format_date(date)
        file_name = '_'.join([self._filename_root, polarization, date, motion])
        file_name = file_name + '.' + self.extension
        return file_name

    def get_path_to_scene(self, motion, polarization, date):
        """Writes full path to scene corresponding to specified motion,
        polarization and date

        Args:
            motion (type): {'ascending', 'descending'}
            polarization (str): {'VH', 'VV'}
            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str

        """
        band_directory = self._format_band_directory(motion, polarization)
        filename = self._write_filename(motion, polarization, date)
        path_to_scene = os.path.join(self.root, band_directory, filename)
        return path_to_scene

    def get_path_to_infos(self, motion, polarization, date):
        raise NotImplementedError("No information provided on Sentinel-1 scenes")
