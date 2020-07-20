import os
from .scene_writer import SceneWriter
from ..utils import convert_date_to_aws_path


class S1SceneWriter(SceneWriter):
    """Extends SceneWriter by handling Sentinel-1 data type and writing directory
    structure. We choose a top-down date/file structure.

    Directory would be structured as :

    Args:
        root (str): root directory where scenes are stored
    """

    def __init__(self, root, extension='tif'):
        super().__init__(root=root, extension=extension)

    def _format_date(self, date):
        """Writes name of date subdirectory given date

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

    def get_path_to_scene(self, date, filename):
        """Writes full path to scene corresponding to specified orbit,
        polarization and date

        Args:
            date (str): date formatted as yyyy-mm-dd
            filename (str): name of file to write in

        Returns:
            type: str
        """
        date_directory = self._format_date(date)
        filename = self._format_filename(filename, date)
        path_to_scene = os.path.join(self.root, date_directory, filename)
        return path_to_scene

    def get_path_to_infos(self, date):
        raise NotImplementedError("No information provided on Sentinel-1 scenes")
