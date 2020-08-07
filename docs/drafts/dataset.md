### Patch extraction :
- MODIS and Landsat-8 frames
- Landsat bands : B4, B5, B1, B3
- MODIS bands : B1, B2, B3, B4
- From 2018 at 5 distinct dates --> to be updated with extended dataset

- Reproject to same CRS
- Align by applying bilinear upsampling to MODIS frames to bring to same resolution than Landsat frames
- Use quality assessment maps to discard corrupted image regions
- Extract 256x256 coregistered clean MODIS and Landsat patches
- How many patch location ? How many patches in total ? --> answer in dataset split

### Dataset split:
- We map Landsat patches to their the MODIS and Landsat patches at the same location but next time step
- We split the dataset into X, X and X patches locations for training, validation and testing.
- The resulting sets are respectively composed of X, X and X patches.


_Showcase some pairs of patches_


### Preprocessing:
- We aim to predict reflectance values which are physically sound measurement
- Normalization would entail interband information loss
- We hence prefer using raw pixel values
