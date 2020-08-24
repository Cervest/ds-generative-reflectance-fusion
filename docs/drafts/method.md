# Notations

- Let $L_t, M_t\in\RR^{B\times W\time H}$ respectively denote a pair of Landsat and a MODIS co-registered images at time step $t$, where $B, W, H$ are respectively the number of spectral bands, width and height of images.

- In the following, we address the task of predicting a Landsat-like image $L_{t_p}$ at date $t_p$ given MODIS image at the same date $M_{t_p}$ and last known Landsat image $L_{t_{p-1}}$.  

- Although we use Landsat and MODIS reflectance as in STARFM~\cite{STARFM}, the high spatial resolution and high temporal frequency fusing rationale remains compatible with other remote sensing products.


# Conditional GANs

- cGANs are generative models learning a mapping from random noise $z$ conditioned on a input $c$ to an output $y$ as $y = G(z, c)$. The generator attempts to fool a discriminator which outputs a scalar $D(y, c)$ to discriminate real from $G$-generated samples, conditionned on $c$. The generator and the discriminator engage in a minimax two-player game.

- GANs correspond to a minimax two-player game where a stochastic generator $G$ attempts to learn the training data distribution and produce synthetic samples to fool a discriminator $D$, itself adversarially trained to discriminate real from synthetic samples.

- Let $x\in\cX$ denote an input image and $y\in\cY$ and output image.

- In the cGAN framework, the generator $G$ follows a conditional law dictated by its input sample $x\in\cX$. $G(x)$ is then a stochastic process valued in the output space $\cY$. Stochasticity of $G(x)$ is limited to dropout layers in this work.

- Similarly, the discriminator $D = \cY\times\cX\rightarrow\{0, 1\}$ also observes the conditionning input $x$ when classifying generated samples $y\sim G(x)$.

- The objective of a cGAN writes :

\begin{equation}
\min_G\max_D = \EE_{x, y}[\log\, D(x, y)] + \EE_{x}[\log\, D(x, G(x))]
\end{equation}


# Deep Reflection Prediction by Fusion

- Let $\cD = \{(L_{t_i}, M_{t_i})\}_{i=1}^n$ a time serie of Landsat and MODIS images. For the sake of simplicity, we will suppose images are over the same site.

- For a given time step $t_i$, suppose we do not have access to Landsat surface reflectance $L_{t_i}$ and would like to predict it.

- This operation requires, on the one hand, precise structural information about ground-level instances which we can obtain from the last known Landsat image $L_{t_{i-1}}$, and on the other hand, reflectance information at prediction date which we can get at coarse resolution from MODIS image $M_{t_i}$.

- Let $I_{t_i} = [L_{t_{i-1}}, M_{t_i}]$ denote the concatenation of both images.

- We propose to frame this problem within cGANs framework and train a generator to estimate $\hat L_{t_i} = G(z, I_{t_i})$.  

- Applying the cGAN framework to this task, we derive from \ref{label:cgan_loss} an adversarial formulation that writes

\begin{equation}
\cL_{\text{cGAN}}(G, D) = \EE[D(L_{t_i}, I_{t_i})] + \EE[D(G(I_{t_i}), I_{t_i})]
\end{equation}

where the expectation is taken over all possible $(I_{t_i}, L_{t_i})$ and stochastic process $G(I_{t_i})$.

- To additionally constraint $G$ to output images close to $L_{t_i}$, we augment the objective with an $L_1$-supervision loss given by:

\begin{equation}
\cL_{L_1}(G) = \EE[\|L_{t_i} - G(I_{t_i})\|_1]
\end{equation}

- $L_1$ is chosen over $L_2$ as in induces less blurring.

- Eventually, the generative reflection prediction by fusion objective is formulated as

\begin{equation}
\min_G\max_D \cL_{\text{cGAN}}(G, D) + \lambda \cL_{L_1}(G)
\end{equation}

- where $\lambda > 0$ is the supervision weight hyperparameter.

# Dataset

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
