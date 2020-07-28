from .scene_writer import SceneWriter
from ..format import AWSFormatter


class S1SceneWriter(AWSFormatter, SceneWriter):
    """Extends SceneWriter by handling Sentinel-1 data type and writing directory
    structure. We choose a top-down date/file structure.

    Directory would be structured as :

    Args:
        root (str): root directory where scenes are stored
    """

    def __init__(self, root, extension='tif'):
        super().__init__(root=root, extension=extension)

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

    def get_path_to_infos(self, date):
        raise NotImplementedError("No information provided on Sentinel-1 scenes")
