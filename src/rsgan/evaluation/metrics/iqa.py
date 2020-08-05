import numpy as np
from scipy import signal
from skimage import metrics


def psnr(ref, tgt):
    """Computes peak signal to noise ratio of a target image wrt to a
    reference image

    Args:
        ref (np.ndarray): Reference image as (height, width)
        tgt (np.ndarray): Target image as (height, width)

    Returns:
        type: float

    """
    return metrics.peak_signal_noise_ratio(image_true=ref,
                                           image_test=tgt)


def ssim(ref, tgt, window_size=None):
    """Computes mean structural similarity index between two images

    "Image quality assessment: from error visibility to structural similarity",
    Wang et. al 2004

    Args:
        ref (np.ndarray): Reference image as (height, width)
        tgt (np.ndarray): Target image as (height, width)
        window_size (int): side-length of the sliding window used in comparison
            must be odd value (default: None)

    Returns:
        type: float

    """
    return metrics.structural_similarity(im1=ref,
                                         im2=tgt,
                                         win_size=window_size)


def sam(ref, tgt, reduce=None):
    """Computes normalized Spectrale Angle Mapper as introduced in

    "Discrimination among semi-arid landscape endmembers using the spectral
    angle mapper (SAM) algorithm", Boardman et al. 1993

    Args:
        ref (np.ndarray): Reference image as (height, width, channels)
        tgt (np.ndarray): Target image as (height, width, channels)
        reduce (str): output reduction method

    Returns:
        type: {np.ndarray, float}
    """
    # Compute pixelwise bands inner product
    kernel = np.einsum('...k,...k', ref, tgt)

    # Normalize inner products
    square_norm_ref = np.einsum('...k,...k', ref, ref).clip(min=np.finfo(np.float16).eps)
    square_norm_tgt = np.einsum('...k,...k', tgt, tgt).clip(min=np.finfo(np.float16).eps)
    normalized_kernel = kernel / np.sqrt(square_norm_ref * square_norm_tgt)

    # Convert to angles
    normalized_angles = np.arccos(normalized_kernel.clip(min=-1, max=1)) / np.pi
    if not reduce:
        output = normalized_angles
    elif reduce == 'mean':
        output = normalized_angles.mean()
    else:
        raise ValueError("Unknown reduce method")
    return output


def cw_ssim(ref, tgt, width=30, K=0.01):
    """Compute the complex wavelet SSIM (CW-SSIM) value from the reference
    image to the target image.

    "CW-SSIM based image classification", Gao et. al 2011

    Code based on https://github.com/jterrace/pyssim/blob/master/ssim/ssimlib.py

    Args:
      ref (np.ndarray): Reference image as (height, width)
      tgt (np.ndarray): Target image as (height, width)
      width (int): width for the wavelet convolution (default: 30)
      K (float): small algorithm parameter (default: 0.01)
    Returns:
      type: float
    """
    # Define a width for the wavelet convolution
    widths = np.arange(1, width + 1)

    # Convolution
    cwt1 = signal.cwt(ref.flatten(), signal.ricker, widths)
    cwt2 = signal.cwt(tgt.flatten(), signal.ricker, widths)

    # Compute the first term
    c1c2 = np.abs(cwt1) * np.abs(cwt2)
    c1_2 = np.square(np.abs(cwt1))
    c2_2 = np.square(np.abs(cwt2))
    numerator_ssim1 = 2 * c1c2.sum(axis=0) + K
    denominator_ssim1 = c1_2.sum(axis=0) + c2_2.sum(axis=0) + K

    # Compute the second term
    c1c2_conj = cwt1 * np.conjugate(cwt2)
    numerator_ssim2 = 2 * np.abs(c1c2_conj.sum(axis=0)) + K
    denominator_ssim2 = 2 * np.abs(c1c2_conj).sum(axis=0) + K

    # Construct the result
    ssim_map = (numerator_ssim1 / denominator_ssim1) * (numerator_ssim2 / denominator_ssim2)

    # Average the per pixel results
    index = np.mean(ssim_map)
    return index
