### MODIS and Landsat-8 datasets
- The study area is in the department of Marne, within the Grand Est region of France
- It is mostly constituted of crops, the rest being either forest or urban surface
- We acquire Landsat-8 30m imagery for 14 dates between 2013 and 2020 along with MODIS 500m surface reflectance product (MCD43) at the same dates
- Images are reprojected and MODIS frames are resampled to the same resolution and bounds than Landsat with bilinear upsampling, all images thus having same size and coordinate system
- We limit spectral domain to red, near infrared (NIR), blue and green bands


### Patch extraction :
- Use quality assessment maps to discard contaminated image regions
- Extract $(256, 256)$ non-overlapping registered Landsat and MODIS patches at each date
- Resulting in 548 patch location and 5671 Landsat-MODIS pairs samples

### Dataset split:
- We split the dataset into 383, 82 and 83 patches locations for training, validation and testing
- The resulting sets are respectively composed of 3526, 801 and 796 samples


_Showcase some pairs of patches_


### Preprocessing:
- We aim to predict reflectance values which are physically sound measurement
- Normalization would entail interband information loss
- We hence prefer using raw pixel values


### Appendix
- Bands : 4-5-1-3 for Landsat and 1-2-3-4 for MODIS + add central wavelength
- List of acquisition dates
