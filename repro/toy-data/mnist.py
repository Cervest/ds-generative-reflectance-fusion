import os
from torchvision.datasets import MNIST


def download_and_extract():
    """
    Downloads and extracts data on the fly
    :return:
    """
    data_dir = 'data/mnist'
    os.makedirs(data_dir, exist_ok=True)
    dataset = MNIST(root=data_dir, download=True)
    del dataset


# import subprocess
# import os
# import shutil

# def download_and_extract():
#     """
#     Downloads and extracts data on the fly
#     :return:
#     """
#     os.makedirs("data/mnist", exist_ok=True)
#     for file in ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]:
#         p = subprocess.Popen("wget http://yann.lecun.com/exdb/mnist/{}.gz".format(file), shell=True)
#         p.wait()
#         p = subprocess.Popen("gzip -d {}.gz".format(file), shell=True)
#         p.wait()
#         shutil.move(file, "data/mnist/{}".format(file))

if __name__ == "__main__":
    download_and_extract()
