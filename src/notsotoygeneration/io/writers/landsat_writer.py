from .scene_writer import SceneWriter
from ..format import LandsatAWSFormatter


class LandsatSceneWriter(LandsatAWSFormatter, SceneWriter):
    """Extends SceneReader by handling Landsat directory structure

    For example, scenes directories would usually be structured as :

        root
        └── 197026                  # Landsat WRS coordinate
                └── 2018            # Year
                    └── 01          # Month
                        └── 01      # Day
                            └── 0/  # Snapshot id

    where each subdirectory has substructure :

        0/
        └── 197026_2018-01-21_0.tif

    Landsat 8 informations : https://www.usgs.gov/land-resources/nli/landsat/landsat-8

    Args:
        root (str): root directory where scenes are stored
    """
    def __init__(self, root, extension='tif'):
        super().__init__(root=root, extension=extension)
