# General motivation

- Amid climate variability induced impact on land resource management, Earth observation data has become increasingly important. In particular, decision making on agricultural issues can greatly benefit from airborne crop monitoring.

- Satellites sensors capture terrestrial surface reflectance at various spatiotemporal resolutions.

- In general a trade-off between temporal and spatial resolution is faced. Sensors with large scanning swath cover wide regions at once, resulting in a poor spatial resolution, but tend to have a great revisit frequency. Inversely, sensors with higher spatial resolution can scan tighter regions, providing finer views, but take longer to revisit a given site.

- In order to overcome limited access to free Earth observation with spatial and temporal resolutions suited for land-cover monitoring, it is yet possible to combine images at different resolutions to synthetize mixed products with adequate resolution characteristics.


- Landsat mission in particular, provides imagery at a 30m spatial resolution appropriate for agricultural monitoring tasks such as crop growth monitoring and land-cover change detection. However, its 16-day revisit cycle and frequent cloud occulting hampers its potential for applications.

- On the other hand, Moderate Resolution Imaging Spectroradiometer (MODIS) sensor offers an interesting daily revisit cycle at the expense of a much coarser spatial resolution of 250m for red and near-infrared (NIR) bands (1-2) and 500m for other optical bands 3-7.

- By combining Landsat fine-resolution acquisitions with MODIS coarser daily update, it is possible to construct a product with a Landsat-like spatial granularity and a MODIS-like temporal coverage.

# Introduce STARFM algorithm and build up motivation to port it with deep learning

- The work by Gao et introduces a spatial and temporal adaptative reflectance fusion model (STARFM) to
predict daily surface reflectance at Landsat spatial resolution and MODIS temporal frequency.

- Using one or more Landsat and MODIS frames, they propose to interpolate neighbouring coarse and high resolution pixels according to their spatial, temporal and spectral proximity in order to recover pixels from a Landsat-like frame.

- More recently, deep learning has shown promise when applied to specific tasks on publicly available missions such as Sentinel-1 and 2, MODIS, Landsat. Notably, deep generative models have demonstrated their capacity to combine multiple frames with low spatial resolution frame into a higher resolution one.


# Paper proposition, contribution and outline

- In this paper, we propose to carry deep generative models success to fusing MODIS and Landsat reflectance

- Contributions are ...
