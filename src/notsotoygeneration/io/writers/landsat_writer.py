from .scene_writer import SceneWriter
from ..format import AWSFormatter


class LandsatSceneWriter(AWSFormatter, SceneWriter):
    """Extends SceneReader by handling Landsat data type and directory structure

    For example, directory would usually be structured as :

        root
        └── 197026                  # Landsat WRS coordinate
                └── 2018            # Year
                    └── 01          # Month
                        └── 01      # Day
                            └── 0/  # Snapshot id

    where each subdirectory has substructure :

        0/
        └── 197026_2018-01-01_0.tif

    MODIS Bands informations : https://modis.gsfc.nasa.gov/about/specifications.php

    Args:
        root (str): root directory where scenes are stored
    """
    def __init__(self, root, extension='tif'):
        super().__init__(root=root, extension=extension)

    def _format_location_directory(self, coordinate, *args, **kwargs):
        """Write directory corresponding to coordinates

        Args:
            coordinate (int): WRS Landsat coordinate as 197026 or 198026

        Returns:
            type: str
        """
        return str(coordinate)

    def _get_default_filename(self, coordinate, date, is_quality_map=False):
        """Composes default filename for writing file as concatenation of wrs
        coordinate and date with file extension

        Args:
            coordinate (int): WRS Landsat coordinate as 197026 or 198026
            date (str): date formatted as yyyy-mm-dd
            is_quality_map (bool): True if is quality map

        Returns:
            type: str
        """
        tokens = [str(coordinate), date]
        if is_quality_map:
            tokens += ['QA']
        filename = '_'.join(tokens)
        filename = filename + '.' + self.extension
        return filename

    def get_path_to_infos(self, modis_coordinate, date):
        raise NotImplementedError("No infos sorry")
