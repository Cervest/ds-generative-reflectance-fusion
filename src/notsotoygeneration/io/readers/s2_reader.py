import os
from .scene_reader import SceneReader


class S2BandReader(SceneReader):
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
        return mgrs_coordinate

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

    def _format_filename(self, band):
        """Makes sure name of file to load is properly formatted

        Args:
            band (str): name of band to load (e.g. 'B02', 'B03', etc)

        Returns:
            type: str
        """
        if not band.endswith(self.extension):
            band = '.'.join([band, self.extension])
        return band

    def get_path_to_scene(self, mgrs_coordinate, date, band):
        """Writes full path to scene corresponding to specified mgrs coordinate,
        date and band

        Args:
            mgrs_coordinate (str): coordinate formatted as '31TBF'
            date (str): date formatted as yyyy-mm-dd
            band (str): name of band to load (e.g. 'B02', 'B03', etc)

        Returns:
            type: str
        """
        mgrs_directory = self._format_mgrs_directory(mgrs_coordinate)
        date_directory = self._format_date_directory(date)
        filename = self._format_filename(band, mgrs_coordinate, date)
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
