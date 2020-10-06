import torch
import torch.nn as nn
import torch.nn.functional as F


def heat_1d_kernel(kernel_size, sigma):
    """Creates unidimensional heat kernel tensor

    Args:
        kernel_size (int): tensor size
        sigma (float): gaussian standard deviation

    Returns:
        type: torch.Tensor

    """
    centered_x = torch.linspace(0, kernel_size - 1, kernel_size).sub(kernel_size // 2)
    heat_1d_tensor = torch.exp(- centered_x.pow(2).div(float(2 * sigma**2)))
    heat_1d_tensor.div_(heat_1d_tensor.sum())
    return heat_1d_tensor


def heat_2d_kernel(kernel_size, channels):
    """Creates square bidimensional heat kernel tensor with specified number of
    channels

    Args:
        kernel_size (int): tensor size
        sigma (float): gaussian standard deviation

    Returns:
        type: torch.Tensor

    """
    _1d_window = heat_1d_kernel(kernel_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    kernel = _2d_window.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel


class SSIM(nn.Module):
    """Differential module implementing Structural-Similarity index computation

    "Image quality assessment: from error visibility to structural similarity",
    Wang et. al 2004

    Largely based on the work of https://github.com/Po-Hsun-Su/pytorch-ssim

    Args:
        kernel_size (int): convolutive kernel size
        C1 (float): weak denominator stabilizing constant (default: 0.01 ** 2)
        C2 (float): weak denominator stabilizing constant (default: 0.03 ** 2)
    """
    def __init__(self, kernel_size=11, C1=0.01**2, C2=0.03**2):
        super(SSIM, self).__init__()
        self.kernel_size = kernel_size
        self.channels = 1
        self.kernel = heat_2d_kernel(kernel_size, self.channels)
        self.C1 = C1
        self.C2 = C2

    def _compute_ssim(self, img1, img2, kernel):
        """Computes mean SSIM between two batches of images given convolution kernel

        Args:
            img1 (torch.Tensor): (B, C, H, W)
            img2 (torch.Tensor): (B, C, H, W)
            kernel (torch.Tensor): convolutive kernel used for moments computation

        Returns:
            type: torch.Tensor

        """
        # Retrieve number of channels and padding values
        channels = img1.size(1)
        padding = self.kernel_size // 2

        # Compute means tensors
        mu1 = F.conv2d(input=img1, weight=kernel, padding=padding, groups=channels)
        mu2 = F.conv2d(input=img2, weight=kernel, padding=padding, groups=channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1.mul(mu2)

        # Compute std tensors
        sigma1_sq = F.conv2d(input=img1 * img1, weight=kernel, padding=padding, groups=channels).sub(mu1_sq)
        sigma2_sq = F.conv2d(input=img2 * img2, weight=kernel, padding=padding, groups=channels).sub(mu2_sq)
        sigma12 = F.conv2d(input=img1 * img2, weight=kernel, padding=padding, groups=channels).sub(mu1_mu2)

        # Compute ssim map and return average value
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return ssim_map.mean()

    def forward(self, img1, img2):
        """Computes mean SSIM between two batches of images

        Args:
            img1 (torch.Tensor): (B, C, H, W)
            img2 (torch.Tensor): (B, C, H, W)

        Returns:
            type: torch.Tensor

        """
        # If needed, recompute convolutive kernel
        channels = img1.size(1)
        if channels == self.channels and self.kernel.data.type() == img1.data.type():
            kernel = self.kernel
        else:
            kernel = heat_2d_kernel(self.kernel_size, channels)
            kernel = kernel.to(img1.device).type_as(img1)
            self.kernel = kernel
            self.channels = channels

        # Compute mean ssim
        ssim = self._compute_ssim(img1, img2, kernel)
        return ssim
