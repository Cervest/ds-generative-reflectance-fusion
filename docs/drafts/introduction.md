# General motivation

- Amid climate variability induced impact on land resource management, Earth observation data has become increasingly important. In particular, decision making on agricultural matters can greatly benefit from airborne crop monitoring.

- Satellites sensors capture terrestrial surface reflectance at various spatiotemporal resolutions.

- In general a trade-off between temporal and spatial resolution is faced. Sensors with large scanning swath cover wide regions at once, resulting in a poor spatial resolution, but tend to have a great revisit frequency. Inversely, sensors with higher spatial resolution can scan tighter regions, providing finer views, but take longer to revisit.

- In order to overcome limited access to free Earth observation with spatial and temporal resolutions suited for land-cover monitoring, it is yet possible to combine images at different resolutions to synthetize mixed products with adequate resolution characteristics to assist decision makers with detailed information in a timely manner.

- Landsat mission in particular, provides imagery at a 30m spatial resolution appropriate for agricultural monitoring tasks such as crop growth monitoring and land-cover change detection. However, its 16-day revisit cycle and frequent cloud occulting hampers its use for monitoring phenomena with rapid transition.

- On the other hand, Moderate Resolution Imaging Spectroradiometer (MODIS) sensor offers an interesting daily revisit cycle at the expense of a much coarser spatial resolution of 250m for red and near-infrared (NIR) bands (1-2) and 500m for other optical bands 3-7.

- The complementarity of Landsat fine-resolution acquisitions and MODIS coarser daily update makes it possible to construct a product with a Landsat-like spatial resolution and a MODIS-like temporal coverage.

# Introduce data fusion and build up motivation to port it with deep learning

- Statistical data fusion techniques have been developed to combine images captured by sensors with different characteristics. The generally assume consistency in reflectance between low and high resolution data, which has notably been pointed out between MODIS and Landsat imagery.

- However, the prediction faithfulness heavily relies on the number of input images and their temporal proximity to the prediction date.

- More recently, deep learning has shown promise when applied to specific tasks on publicly available missions such as Sentinel-1 and 2, MODIS, Landsat. Notably, deep generative models have demonstrated their capacity to combine multiple frames with low spatial resolution frame into a higher resolution one.


# Paper proposition, contribution and outline

- In this paper, we propose a deep learning based reflectance fusing method for fusing Landsat and MODIS imagery.

- Contributions are :
  - Create a dataset of paired Landsat and MODIS time series
  - Introduce a deep generative framework to estimate Landsat-like images by fusing the Landsat reflectance at closest date and MODIS information at target prediction date
  - Evaluate 
