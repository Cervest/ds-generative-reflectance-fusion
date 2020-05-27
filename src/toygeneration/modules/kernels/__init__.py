from .kernels import *
from sklearn.gaussian_process import kernels
from src.utils import Registry
"""
Registery of common kernels
"""
KERNELS = Registry()


def build_kernel(cfg):
    return KERNELS[cfg['name']](cfg)


@KERNELS.register('rbf')
def build_rbf_kernel(cfg):
    kernel = kernels.RBF(length_scale=cfg['length_scale'])
    return kernel


@KERNELS.register('rational_quadratic')
def build_rational_quadratic_kernel(cfg):
    kernel = kernels.RationalQuadratic(length_scale=cfg['length_scale'],
                                       alpha=cfg['alpha'])
    return kernel


@KERNELS.register('sin_squared')
def build_exp_sin_squared_kernel(cfg):
    kernel = kernels.ExpSineSquared(length_scale=cfg['length_scale'],
                                    periodicity=cfg['periodicity'])
    return kernel


@KERNELS.register('constant')
def build_constant_kernel(cfg):
    kernel = kernels.ConstantKernel(constant_value=cfg['constant_value'])
    return kernel
