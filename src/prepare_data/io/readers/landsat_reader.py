import os
import re
from .scene_reader import SceneReader, BandReader
from ..format import LandsatAWSFormatter


class LandsatSceneReader(LandsatAWSFormatter, SceneReader):
    """Extends SceneReader by handling Landsat scenes directory structure

    For example, directory would usually be structured as :

        root
        └── 197026                  # Landsat WRS coordinate
                └── 2018            # Year
                    └── 01          # Month
                        └── 21      # Day
                            └── 0/  # Snapshot id

    where each subdirectory has substructure :

        0/
        └── 197026_2018-01-21_0.tif

    Args:
        root (str): root directory where scenes are stored
    """
    def __init__(self, root, extension='tif'):
        super().__init__(root=root, extension=extension)


class LandsatBandReader(BandReader):
    """Extends band reader by handling Landsat raw bands directory structure

    Typically structured as :

    root
    └── LC081970262018071901T1-SC20200408131922
        ├── LC08_L1TP_197026_20180719_20180731_01_T1_pixel_qa.tif
        ├── LC08_L1TP_197026_20180719_20180731_01_T1_sr_band1.tif
        ├── ...
        └── LC08_L1TP_197026_20180719_20180731_01_T1_sr_band7.tif

    Args:
        root (str): root directory where scenes are stored

    Landsat 8 informations : https://www.usgs.gov/land-resources/nli/landsat/landsat-8
    """

    def __init__(self, root):
        self.root = root
        self.root_directories = os.listdir(root)

    @staticmethod
    def filter_on_regexp(strings, pattern):
        """Given an iterable of strings, filters all strings that do not
        contain provided pattern

        Args:
            strings (list[str])
            pattern (str)

        Returns:
            type: iter[str]
        """
        filtered_strings = filter(lambda x: re.search(pattern, x), strings)
        return filtered_strings

    def get_path_to_scene(self, coordinate, date, filename):
        """Writes path to scene by filtering directories on coordinate and date
            and then directory files on filename

        Args:
            coordinate (int): WRS Landsat coordinate as 197026 or 198026
            date (str): date formatted as yyyy-mm-dd
            filename (str): substring of name of file to read from

        Returns:
            type: str

        """
        try:
            # Filter list of root directories based on coordinate and date
            buffer = self.filter_on_regexp(self.root_directories, str(coordinate))
            buffer = self.filter_on_regexp(buffer, date.replace('-', ''))

            # Join to absolute scene directory path
            scene_directory = next(buffer)
            scene_directory_path = os.path.join(self.root, scene_directory)

            # Filter scenes based on filename substring
            scene_files = os.listdir(scene_directory_path)
            buffer = self.filter_on_regexp(scene_files, filename)

            # Join to absolute path to scene and return
            filename = next(buffer)
            path_to_scene = os.path.join(scene_directory_path, filename)
            return path_to_scene
        except StopIteration:
            raise FileNotFoundError(f"No Landsat file corresponding to specified arguments")

    def get_path_to_infos(self, *args, **kwargs):
        raise NotImplementedError
