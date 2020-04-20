# A Toy Virtual Dataset for Remote Sensing Imagery Problems Evaluation
Toy experimental setup for virtual remote sensing feasibility testing


## Getting Started

_provide very short code sample and result to give a grasp of what the project is about and what's achievable with the code (ideally some quick visualization)_

<img src="https://github.com/Cervest/ds-virtual-remote-sensing-toy/blob/master/docs/source/img/mnist_ideal_generation.png" alt="just to have an image" width="650"/>

```python
# Insert some code to give glimpse of how it works / what you can do
```


## Overview

### Motivation

In the framework of remote sensing, generative models are of keen interest as they could help cope with limited access to costly high-resolution imagery through super-resolution models. We believe, that from a general tool to combine observations from different missions into a higher-resolution one will be built using neural network based generative modeling.

However, when it come to the evaluation of generative methods previous work has been mostly relying on qualitative evidence and some quantitative metrics often poorly consistent between each other. We propose in this work a new synthetic toy dataset structured as :

- An ideal latent high-resolution toy imagery product
- Mutliple lower-resolution products derived from the ideal one

A super-resolution task can then be stated from the point of view of missing data wrt the latent product and having access to such synthetic groundtruth allows to accurately quantify the discrepancy with the results generated out of several coarser observations.

### Organization

The repository is structured as follows :

```
├── data
├── docs
├── notebooks
├── src
├── tests
├── utils
├── environment.yml
└── README.md
```
- `data` : datasets used for toy dataset generation process
- `docs`: any paper, notes, image relevant to this repository
- `notebooks`: presentation and example notebooks
- `src`: contains all modules and scripts to reproduce generation process
- `tests`: unit testing modules
- `utils`: miscellaneous utilities

### Features description
_Add functionalities presentation (links for more details over functionalities in [wiki](https://github.com/Cervest/ds-virtual-remote-sensing-toy/wiki))_

## Installation

All code is implemented in Python 3.7

_TODO : Add additional explanation steps for installation_

#### Setting up environment
```bash
$ git clone https://github.com/Cervest/ds-virtual-remote-sensing-toy.git
$ cd ds-virtual-remote-sensing-toy
$ conda env create -f environment.yml
$ conda activate toy-vrs
```

#### Setting up dvc

From the environment and root project directory :

```bash
$ (toy-vrs) dvc init -q
$ (toy-vrs) python repro/dvc.py --link=where/data/will/be/stored
$ (toy-vrs) dvc repro repro/toy-data/download_data.dvc
```

In case pipeline dvc file is missing, one can recreate it by running :

```bash
$ ./repro/toy-data/.setup_data.sh
```


## References
