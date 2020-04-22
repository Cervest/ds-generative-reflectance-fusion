import subprocess
import os


def download_and_extract():
    """
    Downloads and extracts data on the fly
    :return:
    """
    os.makedirs("data/ts", exist_ok=True)
    for file in ["Multivariate2018_arff", "Multivariate2018_ts"]:
        p = subprocess.Popen("wget -qO- http://www.timeseriesclassification.com/Downloads/Archives/{}.zip | bsdtar -xvf- -C data/time-series/".format(file), shell=True)
        p.wait()


if __name__ == "__main__":
    download_and_extract()
