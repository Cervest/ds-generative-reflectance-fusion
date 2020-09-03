from .scene_writer import SceneWriter
from ..format import MODISAWSFormatter


class MODISSceneWriter(MODISAWSFormatter, SceneWriter):
    """Extends SceneWriter by handling MODIS directory structure

    For example, directory would usually be structured as :

        root
        └── 18                          # MODIS horizontal tile
            └── 04                      # MODIS vertical tile
                └── 2017                # Year
                    └── 1               # Month
                        ├── 1           # Day
                        │   └── 0/
                        │  ...
                        └── 31
                            └── 0/

    where each subdirectory has substructure :

        0/
        ├── scene.tif      # Scene file (one or more)
        └── infos.json     # Infos file

    MODIS informations : https://modis.gsfc.nasa.gov/about/specifications.php

    Args:
        root (str): root directory where scenes are stored
    """
    def __init__(self, root, extension='tif'):
        super().__init__(root=root, extension=extension)
