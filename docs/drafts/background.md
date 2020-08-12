# Reflectance fusion models

- Spatial and temporal adaptative reflectance fusion model (STARFM) introduced to
predict daily surface reflectance at Landsat spatial resolution and MODIS temporal
frequency
- Propose to interpolate neighbouring coarse-resolution pixels weighted by their spatial, spectral and temporal proximity to a fine-resolution resolution pixel to recover

- Have entailed numerous applications \cite{all_STARM_stuff}

- However rely and handcrafted heuristics involving consistency assumptions


# Generative imagery in remote sensing

- Multiple studies have presented compelling results using deep generative models for diverse remote sensing tasks such as cloud removal from optical imagery, SAR to optical image translation or spatial resolution enhancement.

- Notably, multi-frame super-resolution works by \cite{highresnet, deepsum} in the course of the PROBA-V competition have shed light on neural networks ability to fuse information from low spatial resolutions frames to reconstruct an image with enhanced granularity.

- GAN based models, which have demonstrated excellent performance in natural image generation~\cite{pix2pix}, have also shown promising performance in generating realistic synthetic remote sensing imagery \cite{grohnfeldt, wang, reyes}.
