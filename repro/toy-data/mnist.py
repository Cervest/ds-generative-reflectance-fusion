import os
from torchvision.datasets import MNIST


def download_and_extract():
    """
    Downloads and extracts data on the fly
    :return:
    """
    data_dir = "data/mnist"
    os.makedirs(data_dir, exist_ok=True)
    dataset = MNIST(root=data_dir, download=True)
    del dataset


if __name__ == "__main__":
    download_and_extract()
