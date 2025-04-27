import abc
import torch

from .base import BaseLinearModelWithGaussianNoise

from mean_system_response.downsample import DownsampleMeanSystemResponse
from noise_covariance.scalar import ScalarNoiseCovariance

class DownsampleModelWithScalarCovarianceNoise(BaseLinearModelWithGaussianNoise):
    '''
    Linear inverse problem composed of downsampling operator (A) and scalar variance (std^2).

    y = Ax + std \epsilon,
    '''


    def __init__(self, scale_factor: int|list[int], var: float):
        super(DownsampleModelWithScalarCovarianceNoise, self).__init__()
        self.mean_system_response = DownsampleMeanSystemResponse(scale_factor)
        self.noise_covariance = ScalarNoiseCovariance(var)


    def fix_state(self, *args, **kwargs):
        pass


    def pinvA(self, y):
        return self.mean_system_response.pinvA(y)
    

    def pinvA_A(self, x):
        return self.mean_system_response.pinvA(self.mean_system_response.A(x))
    

    def pinvA_Sigma_sqrt(self, x):
        return self.mean_system_response.pinvA(self.noise_covariance.sqrtSigma(x))