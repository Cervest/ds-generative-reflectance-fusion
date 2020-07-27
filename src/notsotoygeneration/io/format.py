from abc import ABC, abstractmethod
import os
from .utils import convert_date_to_aws_path


class ScenePathFormatter(ABC):
    """Backbone interface defining path formatting methods for access to imagery
    scenes
    """

    def _format_filename(self, filename, *args, **kwargs):
        """Formats name of file to access

        Args:
            filename (str): name of file to access

        Returns:
            type: str
        """
        if not filename:
            self._get_default_filename(*args, **kwargs)
        return filename

    @abstractmethod
    def _get_default_filename(self, *args, **kwargs):
        """Writes filename in default fashion

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

    @abstractmethod
    def _format_location_directory(self, coordinate, *args, **kwargs):
        """Writes location-related directory path

        Returns:
            type: str
        """
        pass


class AWSFormatter(ScenePathFormatter):
    """Handles specificities of AWS-like directory structures"""

    def _format_date_directory(self, date):
        """Writes date-related directory path

            date (str): date formatted as yyyy-mm-dd

        Returns:
            type: str
        """
        date_directory = convert_date_to_aws_path(date)
        return date_directory
