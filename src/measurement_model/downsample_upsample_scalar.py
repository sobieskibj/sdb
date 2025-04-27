import abc
import torch

from .base import BaseLinearModelWithGaussianNoise

from mean_system_response.downsample_upsample import DownsampleUpsampleMeanSystemResponse
from noise_covariance.scalar import ScalarNoiseCovariance

class DownsampleUpsampleModelWithScalarCovarianceNoise(BaseLinearModelWithGaussianNoise):
    '''
    Linear inverse problem composed of downsampling operator (A) and scalar variance (std^2).

    y = Ax + std \epsilon,
    '''


    def __init__(self, scale_factor: int|list[int], upsample_A: bool, var: float):
        super(DownsampleUpsampleModelWithScalarCovarianceNoise, self).__init__()
        self.mean_system_response = DownsampleUpsampleMeanSystemResponse(scale_factor, upsample_A)
        self.noise_covariance = ScalarNoiseCovariance(var)


    def fix_state(self, *args, **kwargs):
        pass