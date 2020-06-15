# A Toy Virtual Dataset for Remote Sensing Imagery Problems Evaluation
Toy experimental setup for virtual remote sensing feasibility testing


## Getting Started

This repository allows you to :
- Generate high resolution synthetic toy imagery and degraded versions of it with lower spatial, temporal resolution and artefacts.
- Train and evaluate several remote-sensing image-translation models at : cloud removal, super-resolution, sar-to-optical translation

<p align="center">
<img src="https://github.com/Cervest/ds-virtual-remote-sensing-toy/blob/master/docs/source/img/latent_vs_derived.png" alt="Ideal image and derived coarser one" width="1000"/>
 </p>

__Synthetic imagery generation :__ First, you can setup a YAML configuration file specifying execution. Templates are proposed under `src/toygeneration/config/` directory. Then, from environment run:

```bash
$ (toy-vrs) python run_toy_generation.py --cfg=path/to/generation/config.yaml --o=sandbox/latent_product
Generation |################################| 31/31
$ (toy-vrs) python run_toy_derivation.py --cfg=path/to/derivation/config.yaml --o=sandbox/derived_product
Derivation |#################               | 16/31
```

For generation as for derivation, created image frames have a corresponding annotation mask for segmentation and classification tasks. Explicitely, output directories are structured as :
```
 ├── frames/           # 1 frame = 1 time step
 │   ├── frame_0.h5
 │   ├── ...
 │   └── frame_31.h5
 ├── annotations/      # frames associated annotation masks
 │   ├── annotation_0.h5
 │   ├── ...
 │   └── annotation_31.h5
 └── index.json
 ```

<p align="center">
<img src="https://github.com/Cervest/ds-virtual-remote-sensing-toy/blob/master/docs/source/img/latent_product.png" alt="Ideal product and annotation masks" width="700"/>
</p>


__Image translation model training and evaluation :__ Similarly, you need to setup a YAML configutation file specifying the experiment execution. Templates are proposed under `src/rsgan/config` directory. Then, to execute training/testing on GPU 0, run:

```bash
$ (toy-vrs) python run_training.py --cfg=path/to/experiment/config.yaml --o=sandbox/my_experiment_logs --device=0
$ (toy-vrs) python run_testing.py --cfg=path/to/experiment/config.yaml --o=sandbox/my_experiment_logs --device=0
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
├── make_baseline_classifier.py
├── run_training.py
├── run_testing.py
├── run_toy_generation.py
└── run_toy_derivation.py
```

__Directories :__
- `data/` : Time series datasets used for toy product generation, generated toy datasets and experiments outputs
- `docs/`: any paper, notes, image relevant to this repository
- `notebooks/`: demonstration notebooks
- `src/`: all modules to run synthetic data generation and experiments
- `tests/`: unit testing
- `utils/`: miscellaneous utilities

---

__`src/` directory is then subdivided into :__

- Synthetic data generation :
```
.
└── toygeneration
    ├── config/
    ├── blob/
    ├── timeserie/
    ├── modules/
    ├── derivation.py
    ├── export.py
    └── product.py
```
- `config/`: YAML configuration specification files for generation and derivation
- `blob/`: Modules implementing blobs such as voronoi polygons which are used in synthetic product
- `timeserie/`: Modules implementing time series handling to animate blobs
- `modules/`: Additional modules used for aggregation, randomization of pixels, polygons computation, degradation effect simulation
- `derivation.py`: Image degradation module
- `export.py`: Synthetic images dumping and loading
- `product.py`: Main synthetic imagery product generation module


- Image-translation experiments :
```
.
└── rsgan/
    ├── config/
    ├── callbacks/
    ├── data/
    │   ├── datasets
    │   └── transforms
    ├── evaluation
    │   └── metrics/
    ├── experiments
    │   ├── cloud_removal/
    │   ├── sar_to_optical/
    │   ├── super_resolution/
    │   ├── experiment.py
    │   └── utils
    ├── losses
    └── models
```
- `config/`: YAML configuration specification files for training and testing of models
- `callbacks/`: Experiment execution callback modules
- `data/`: Modules for used imagery datasets loading
- `evaluation/`: Misc useful modules for evaluation
- `experiments/`: Main directory proposing classes encapsulating each experiment
- `losses`: Losses computation modules
- `models`: Neural networks models used in experiments


### Features description
_Add functionalities presentation (links for more details over functionalities in [wiki](https://github.com/Cervest/ds-virtual-remote-sensing-toy/wiki))_

## Installation

Code implemented in Python 3.8

#### Setting up environment

Clone and go to repository
```bash
$ git clone https://github.com/Cervest/ds-virtual-remote-sensing-toy.git
$ cd ds-virtual-remote-sensing-toy
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
$ (toy-vrs) python repro/dvc.py --link=where/data/stored --cache=where/cache/stored
```
if no `link` specified, data will be stored by default into `data/` directory and fefault cache is `.dvc/cache`.

To download datasets, then simply run:
```bash
$ (toy-vrs) dvc repro
```
In case pipeline is broken, hidden bash files are provided under `repro/downloads/ts/.download_ts.sh`

Then, according to experiment you wish to run, dvc files are gathered `repro/toy-data` to reproduce synthetic dataset generation, and experiment dvc files are proposed under `repro/experiments`
## References
