# Reflectance fusion models

- The work by Gao et introduces a spatial and temporal adaptative reflectance fusion model (STARFM) to
predict daily surface reflectance at Landsat spatial resolution and MODIS temporal frequency using Landsat and MODIS 500m data.

- Let $(M_{t_k}, L_{t_k})_{1\leq k\leq n}$ denote n paired MODIS and Landsat surface reflectance at dates $t_k$ and $M_{t_p}$ a MODIS frame at some prediction date $t_p$. MODIS images have been georeferenced and resampled to Landsat resolution.

- Assuming constant bias between Landsat and MODIS acquisitions, their reflectance consistency entails that for homogeneous pixels, one has $L_{t_p} = M_{t_p} + L_{t_k} - M_{t_k}$. Then, in order to further account for mixed land-cover types and their possible evolution between acquisition dates, Gao et al. extend the spatiotemporal interpolation range.

- Namely, they propose to predict Landsat-like pixels at $t_p$ by interpolating neighbouring pixels within a window of size $w$ as :

$$
L_{t_p}(\nicefrac{w}{2}, \nicefrac{w}{2}) = \sum_{i, j = 1}^{w}\sum_{k=1}^n W_{ijk}\left[M_{t_p}(i, j) + L_{t_k}(i, j) - M_{t_k}(i, j)\right]
$$
- where weight $W_{ijk}$ is based on spatial, temporal and spectral proximity measures of pixels.

- Since then, STARFM has been successfully applied for vegeation monitoring and variations of this algorithm have been proposed to cope with cloud contamination~\cite{ESTARFM} or heterogeneous fined grained areas~\cite{ESTARM, USTARFM}.


# Generative imagery in remote sensing

- Multiple studies have presented compelling results using deep generative models for diverse remote sensing tasks such as cloud removal from optical imagery, SAR to optical image translation or spatial resolution enhancement.

- Notably, multi-frame super-resolution works by \cite{highresnet, deepsum} in the course of the PROBA-V competition have shed light on neural networks ability to fuse information from low spatial resolutions frames to reconstruct an image with enhanced granularity.

- GAN based models, which have demonstrated excellent performance in natural image generation~\cite{pix2pix}, have also shown promising performance in generating realistic synthetic remote sensing imagery \cite{grohnfeldt, wang, reyes}.
