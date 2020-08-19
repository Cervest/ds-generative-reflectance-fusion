# Reflectance fusion models

- The work by Gao et introduces a spatial and temporal adaptative reflectance fusion model (STARFM) to
predict daily surface reflectance at Landsat spatial resolution and MODIS temporal frequency using Landsat and MODIS 500m data.

- Let $(M_{t_k}, L_{t_k})_{1\leq k\leq n}$ denote n paired MODIS and Landsat surface reflectance at dates $t_k$ and $M_{t_p}$ a MODIS frame at some prediction date $t_p$. MODIS images have been georeferenced and resampled to Landsat resolution.

- Assuming constant bias between Landsat and MODIS acquisitions, their reflectance consistency entails that for homogeneous pixels, one has $L_{t_p} = M_{t_p} + L_{t_k} - M_{t_k}$. Then, in order to further account for heterogeneous land-cover and their possible evolution between acquisition dates, Gao et al. extend the spatiotemporal interpolation range.

- Namely, they propose to predict Landsat-like pixels at $t_p$ by interpolating neighbouring pixels within a window of size $w$ as :

$$
L_{t_p}(\nicefrac{w}{2}, \nicefrac{w}{2}) = \sum_{i, j = 1}^{w}\sum_{k=1}^n W_{ijk}\left[M_{t_p}(i, j) + L_{t_k}(i, j) - M_{t_k}(i, j)\right]
$$
- where weight $W_{ijk}$ is based on spatial, temporal and spectral proximity measures of pixels.

- STARFM has been successfully applied for to produce daily surface reflectance at Landsat resolution~\cite{STARFM_NDVI, STARFM_dryland, STARFM_vegetation_monitoring, STARFM_dense_ts_generation}. Improved versions of this algorithm such as Enhanced STARFM (ESTARFM)~\cite{ESTARM} and Unmixing STARFM (USTARFM)~\cite{USTARFM} have been introduced. Based on unmixing theory, they provide consideration to the endmembers of heterogeneous coarse resolution pixels and allow to better cope with  fined grained landscapes and cloud contamination.


# Generative imagery in remote sensing

- Diverse studies have presented compelling results using deep generative models for remote sensing tasks such as cloud removal from optical imagery, SAR to optical image translation or spatial resolution enhancement.

- Notably, Multi-Frame Super-Resolution (MFSR) works by \cite{highresnet, deepsum} in the course of the PROBA-V competition have shed light on neural networks ability to combine information from mutltiple low resolution frames into a synthetic image with finer granularity.

- Rudnet et al. also have shown how fusing products with different characteristics can help better segment flooded buildings by combining complementary temporal, spatial or spectral information.

- Generative Adversarial Networks (GANs) have demonstrated impressive capacity at natural image generation~\cite{brock2018large}, image translation tasks~\cite{pix2pix} and single-smage super-resolution~\cite{berthelot2020creating}, unveiling their ability at extrapolating learnt details from images with poor resolution.

- The conditional GANs (cGANs) rationale has received particular interest in remote sensing as it allows to condition the generated sample on a input. For example, conditionning on SAR images has shown promising performance in generating realistic cloud-free optical remote sensing imagery~\cite{grohnfeldt, wang, reyes}.
