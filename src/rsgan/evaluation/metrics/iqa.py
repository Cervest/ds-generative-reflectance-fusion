import numpy as np
from scipy import signal
from skimage import metrics


def psnr(ref, tgt):
    """Computes peak signal to noise ration of a target image wrt to a
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


def cw_ssim(ref, tgt, width=30, K=0.01):
    """Compute the complex wavelet SSIM (CW-SSIM) value from the reference
    image to the target image.
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
