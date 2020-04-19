import os
import subprocess
from utils.general import read_json

def set_cache(cache_dir):
    """
    Sets dvc cache given config file
    :param cache_dir: path to cache directory
    :return:
    """
    os.makedirs(cache_dir, exist_ok=True)
    p = subprocess.Popen("dvc cache dir {} --local".format(cache_dir), shell=True)
    p.communicate()

def set_symlink():
    p = subprocess.Popen("dvc cache.type symlink --local", shell=True)
    p.communicate()
    p = subprocess.Popen("dvc cache.protected true --local", shell=True)
    p.communicate()


def create_symlink(to_folder, from_folder):
    """
    Creates symlink between local directory and storage directory
    :param to_folder:
    :param from_folder:
    :return:
    """
    os.symlink(from_folder, to_folder)

def main():
    """
    Sets up dvc for large data
    :return:
    """
    set_symlink()
    config = read_json("dvc_config.json")
    cache_dir = config["dvc"]["cache_path"]
    if cache_dir:
        set_cache(cache_dir)
    # Get directory of current project
    cur_project_dir = os.getcwd()
    # Make path to store data
    to_folder = os.path.join(cur_project_dir, "data")
    from_folder = config["dvc"]["symlink_path"]
    os.makedirs(from_folder, exist_ok=True)
    # Create a symlink
    create_symlink(to_folder, from_folder)

if __name__ == "__main__":
    main()