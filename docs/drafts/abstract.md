# Abstract
Mapping land cover changes associated with vegetation dynamics and natural disaster in a timely manner can critically benefit Earth science and humanitarian responses. While satellite imagery stands as a prime asset in this context, remote sensing devices often trade spatial and temporal resolutions due to technical and budget limitations such that no single sensor provides fine grained acquisitions with frequent coverage. In this work, we probe the potential of applying deep generative models to impute high resolution imagery at any date by fusing products with different characteristics. We introduce a dataset of co-registered Moderate Resolution Imaging Spectroradiometer (MODIS) and Landsat surface reflectance time series and benchmark our model against state-of-the-art reflectance fusion algorithms. Our experiments demonstrate the ability to blend in MODIS coarse daily reflectance information into low-paced Landsat detailed acquisitions.


# Proposal
We investigate to what extent can conditional deep generative models manage to blend coarse daily reflectance information provided by MODIS imagery into the fine grained spatial structure of a Landsat image of the same site. Doing so, one would be able to overcome limited access to free Earth observations with high spatial and temporal resolution, which can greatly benefit decision making when it comes to detecting rapid phenomena.

We create a dataset of co-registered surface reflectance time series from MODIS and Landsat - the study area is in the Grand Est region of France - and train conditional generative adversarial networks (cGANs) to predict Landsat-like reflectance in a supervised fashion at any date, given MODIS reflectance at the same date and last known Landsat reflectance. Our cGANs architecture builds upon the success of pix2pix (Isola et al. 2018) in generative remote sensing. We consider two approaches : plain prediction of Landsat-like reflectance and prediction of pixel-wise reflectance difference with last known Landsat image.

Results are then evaluated against a baseline statistical reflectance fusion approach, namely the enhanced spatial and temporal adaptive reflectance fusion algorithm (ESTARFM, Zhu et al. 2010). We compare generated images to groundtruth with full-reference image quality assessment metrics. They point out cGANs ability to capture the gist of MODIS low-resolution reflectance information and fuse it into Landsat spatial structure. However, the model still struggles at predicting details and rendering realistic looking images. In contrast, ESTARFM thrives at producing realistic images, faithful to Landsat spatial structure, but fails to recover righteous reflectance values when input samples are temporally too distant from desired prediction date.


# Machine learning relevance
We use conditional generative adversarial networks to fuse surface reflectance from imagery products with different characteristics.


# Climate change relevance
Access to land cover monitoring instruments with high spatial and temporal resolution would empower decision making amid climate variability induced stress and thus help cope with adaptation to climate change. For example, it would assist response in case of natural disaster such as wildfires, flooding; it would benefit precision agriculture and thus food supply management.


# Additional infos on feedback
We hope to gain insights regarding the relevance of using deep generative modelling to fuse remote sensing products as well as expert advices to further unlock the potential of cGANs and achieve precise Landsat-like reflectance generation.


We use conditional generative adversarial networks to fuse surface reflectance from imagery products with different characteristics.


Access to land cover monitoring instruments with high spatial and temporal resolution would empower decision making amid climate variability induced stress and thus help cope with adaptation to climate change. For example, it would assist response in case of natural disaster such as wildfires, flooding; it would benefit precision agriculture and thus food supply management.


We hope to gain insights regarding the relevance of using deep generative modelling to fuse remote sensing products as well as expert advices to further unlock the potential of cGANs and achieve precise Landsat-like reflectance generation.
