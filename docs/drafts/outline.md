# Introduction



# Background

## Reflectance fusion model

## Generative imagery in remote sensing



# Method

## Dataset

- Images location
- Products used : Landsat, MODIS and bands
- Patch extraction process and results

## Fusing architecture

- Introduce notations and GAN framework
- Generator to predict difference with last known Landsat frame given MODIS frame and last Landsat
- Discriminator on resulting image
- Explicit loss function with supervision
- Early fusion Unet generator amd Patch-GAN discriminator like pix2pix



# Experiment

## Experimental setup

- Dataset split
- Patches preprocessing concerns (no normalization)
- Batch size, optimizers, frequencies, learning rates, supervision weight
- Evaluation metrics

## Reflectance fusing results

- Quantitative table on test set : STARFM / L1 / GAN / L1 + GAN
- Qualitative assessment with visualization
