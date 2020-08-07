import os
import time
import shutil
import json
import pickle
import yaml
import warnings


def mkdir(dirpath, overwrite=False, timestamp=False):
    """
    Creates directory at given location
    Default setting don't allow overwriting if directory already exists
    """
    if timestamp:
        dirpath = dirpath + "_" + time.strftime("%Y%m%d-%H%M%S")
    if os.path.exists(dirpath):
        if overwrite:
            shutil.rmtree(dirpath)
            os.mkdir(dirpath)
        else:
            warnings.warn(f"directory {dirpath} already exists")
    else:
        os.makedirs(dirpath)


def save_json(path, jsonFile):
    """
    Dumps dictionnary as json file
    """
    with open(path, "w") as f:
        f.write(json.dumps(jsonFile))


def load_json(path):
    """
    Loads json format file into python dictionnary
    """
    with open(path, "r") as f:
        jsonFile = json.load(f)
    return jsonFile


def save_pickle(path, file):
    """
    Dumps file as pickle serialized file
    """
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_pickle(path):
    """
    Loads pickle serialized file
    """
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def save_yaml(path, file):
    """Loads YAML file
    """
    with open(path, "w") as f:
        yaml.safe_dump(file, f)
    return file


def load_yaml(path):
    """Loads YAML file
    """
    with open(path, "r") as f:
        file = yaml.safe_load(f)
    return file
