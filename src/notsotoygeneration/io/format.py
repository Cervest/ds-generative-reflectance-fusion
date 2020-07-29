import os
from abc import ABC, abstractmethod
from .utils import convert_date_to_aws_path


class ScenePathFormatter(ABC):
    """Backbone interface defining path formatting methods for access to imagery
    scenes

    Args:
        root (str): root directory where scenes are stored
        extension (str): scenes files extensions {'jp2', 'tif', 'JP2', 'TIF'}

    """
    _scenes_extensions = {'jp2', 'tif'}

    def __init__(self, root, extension):
        self.root = root
        self.extension = extension

    @abstractmethod
    def _format_location_directory(self, coordinate, *args, **kwargs):
        """Writes location-related directory path

        Returns:
            type: str

        """
        pass

    @abstractmethod
    def _format_date_directory(self, date, *args, **kwargs):
        """Writes date-related directory path

        Returns:
            type: str

        """
        pass

    def _format_filename(self, filename, *args, **kwargs):
        """Formats name of file to access

        Args:
            filename (str): name of file to access

        Returns:
            type: str

        """
        if not filename:
            filename = self._get_default_filename(*args, **kwargs)
        return filename

    @abstractmethod
    def _get_default_filename(self, *args, **kwargs):
        """Writes filename in default fashion

        Returns:
            type: str

        """
        pass

    def get_path_to_scene(self, coordinate, date, filename=None, *args, **kwargs):
        """Writes path to scene file as concatenation of location directory,
        date directory and filename

        Returns:
            type: str

        """
        location_directory = self._format_location_directory(coordinate, *args, **kwargs)
        date_directory = self._format_date_directory(date, *args, **kwargs)
        filename = self._format_filename(filename, coordinate, date, *args, **kwargs)
        path_to_scene = os.path.join(self.root, location_directory, date_directory, filename)
        return path_to_scene

    def get_path_to_infos(self, coordinate, date, *args, **kwargs):
        raise NotImplementedError

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

    @property
    def extension(self):
        return self._extension

    @extension.setter
    def extension(self, extension):
        assert extension.lower() in self._scenes_extensions, f"Provided extension {extension} while only {self._scenes_extensions} are allowed"
        self._extension = extension


class AWSFormatter(ScenePathFormatter):
    """Handles specificities of AWS-like directory structures"""

    def _format_date_directory(self, date, *args, **kwargs):
        """Writes date-related directory path

            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        date_directory = convert_date_to_aws_path(date)
        return date_directory
