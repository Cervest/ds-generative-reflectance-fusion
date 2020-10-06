# Deep Generative Reflectance Fusion
Achieving Landsat-like reflectance at any date by fusing Landsat and MODIS surface reflectance with deep generative models.


## Getting Started

<p align="center">
<img src="https://github.com/Cervest/ds-generative-reflectance-fusion/blob/master/docs/source/img/scheme_deep_reflectance_fusion.png" alt="Reflectance Fusion Drawing" width="800"/>
 </p>

### Running experiments

Setup YAML configuration files specifying experiment : dataset, model, optimizer, experiment. See [here](https://github.com/Cervest/ds-generative-reflectance-fusion/tree/master/src/deep_reflectance_fusion/config) for examples.

Execute __training__ on, say GPU 0, as:
```bash
$ python run_training.py --cfg=path/to/config.yaml --o=output/directory --device=0
```

Once training completed, specify model checkpoint to evaluate in previously defined YAML configuration file and run __evaluation__ as:

```bash
$ python run_testing.py --cfg=path/to/config.yaml --o=output/directory --device=0
```

### Preimplemented experiments

| Experiment       | Mean Absolute Error | PSNR | SSIM | SAM |
|------------------|---------------------|------|------|-----|
| ESTARFM          |            -        | 21.0 | 0.645|0.0488|
| [cGAN + L1](https://github.com/Cervest/ds-generative-reflectance-fusion/blob/master/src/deep_reflectance_fusion/config/modis_landsat_fusion/generative/cgan_fusion_unet.yaml)        |        218          | 22.8 | 0.717|0.0275|
| [cGAN + L1 + SSIM](https://github.com/Cervest/ds-generative-reflectance-fusion/blob/master/src/deep_reflectance_fusion/config/modis_landsat_fusion/generative/ssim_cgan_fusion_unet.yaml) |        215          | 23.0 | 0.732|0.0270|


### Compile ESTARFM

To compile ESTARFM please follow [guidelines](https://github.com/Cervest/cuESTARFM#compilation) from official repository.

## Project Structure

```
├── data/
├── repro/
├── src/
│   ├── cuESTARFM
│   ├── deep_reflectance_fusion
│   ├── prepare_data
│   └── utils
├── tests
├── run_training.py
├── run_testing.py
├── run_ESTARFM.py
└── run_ESTARFM_evaluation.py
```

__Directories :__
- `data/` : Landsat-MODIS reflectance time series dataset and experiments outputs
- `repro/`: bash scripts to run data version control pipelines
- `src/`: modules to run reflectance patches extraction and deep reflectance fusion experiments
- `tests/`: unit testing
- `utils/`: miscellaneous utilities


## Installation

Code implemented in Python 3.8

#### Setting up environment

Clone and go to repository
```bash
$ git clone https://github.com/Cervest/ds-generative-reflectance-fusion.git
$ cd ds-generative-reflectance-fusion
```

Create and activate environment
```bash
$ pyenv virtualenv 3.8.2 fusion
$ pyenv activate fusion
$ (fusion)
```

Install dependencies
```bash
$ (fusion) pip install -r requirements.txt
```

#### Setting up dvc

From the environment and root project directory, you first need to build
symlinks to data directories as:
```bash
$ (fusion) dvc init -q
$ (fusion) python repro/dvc.py --link=where/data/stored --cache=where/cache/stored
```
if no `--link` specified, data will be stored by default into `data/` directory and default cache is `.dvc/cache`.

To reproduce a pipeline stage, execute:
```bash
$ (fusion) dvc repro -s stage_name
```
In case pipeline is broken, hidden bash files are provided under `repro` directory

## References
