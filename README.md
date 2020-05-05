# A Toy Virtual Dataset for Remote Sensing Imagery Problems Evaluation
Toy experimental setup for virtual remote sensing feasibility testing


## Getting Started

_provide very short code sample and result to give a grasp of what the project is about and what's achievable with the code (ideally some quick visualization)_

<img src="https://github.com/Cervest/ds-virtual-remote-sensing-toy/blob/master/docs/source/img/latent_vs_derived.png" alt="Ideal image and derived coarser one" width="650"/>

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
├── config/
├── data/
├── docs/
├── notebooks/
├── repro/
├── src/
├── tests/
├── utils/
├── Dvcfile
├── requirements.txt
├── README.md
├── run_generation.py
└── run_derivation.py
```

__Directories :__
- `config/`: YAML configuration specification files for generation and derivation
- `data/` : MNIST and time series datasets used for toy product generation
- `docs/`: any paper, notes, image relevant to this repository
- `notebooks/`: demonstration notebooks
- `src/`: all modules to run generation and derivation process
- `tests/`: unit testing
- `utils/`: miscellaneous utilities


### Features description
_Add functionalities presentation (links for more details over functionalities in [wiki](https://github.com/Cervest/ds-virtual-remote-sensing-toy/wiki))_

## Installation

Code implemented in Python 3.8

#### Setting up environment

Clone and go to repository
```bash
$ git clone https://github.com/Cervest/ds-virtual-remote-sensing-toy.git
$ cd ds-virtual-remote-sensing-toy
$
```

Create and activate environment
```bash
$ pyenv virtualenv 3.8.2 toy-vrs
$ pyenv activate toy-vrs
$ (toy-vrs)
```

Install dependencies
```bash
$ (toy-vrs) pip install -r requirements.txt
```

#### Setting up dvc

From the environment and root project directory, you first need to build
symlinks to data directories as:
```bash
$ (toy-vrs) dvc init -q
$ (toy-vrs) python repro/dvc.py --link=where/data/will/be/stored
```
if no `link` specified, data will be stored by default into `data/` directory.

To download datasets, then simply run:
```bash
$ (toy-vrs) dvc repro
```

In case pipeline is broken, bash files `.download_mnist.sh` and `.download_ts.sh` are provided under `repro/toy-data`.

## References
