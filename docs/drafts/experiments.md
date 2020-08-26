## Experimental setup

### Dataset split:
- We split the dataset into 383, 82 and 83 patches locations for training, validation and testing
- The resulting sets are respectively composed of 3526, 801 and 796 samples

### Preprocessing:
- We aim to predict reflectance values which are physically sound measurement
- Normalization would entail interband information loss
- We hence prefer using raw pixel values

### Training setup
- We use a batch size of 16, supervision weight $\lambda = 5e-2$, Adam optimizer, a learning rate of 2e-3 with exponential decay with factor 0.99
- We update discriminator gradient twice as much as the generator
- We provide quantitative evaluation of generated sample relying on full reference image quality metrics : peak-signal-to-noise ratio (PSNR)~\cite{psnr}, structural similarity (SSIM)~\cite{ssim} and spectral angle mapper (SAM)~\cite{sam}


### Results
