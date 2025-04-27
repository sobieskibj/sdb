import abc
import torch

from .base import BaseLinearModelWithGaussianNoise

from mean_system_response.mask import MaskMeanSystemResponse
from noise_covariance.scalar import ScalarNoiseCovariance


class MaskModelWithNoNoise(BaseLinearModelWithGaussianNoise):
    '''
    Linear inverse problem composed of masking operator (A) and no noise.

    y = Ax,
    '''

    def __init__(self, path_masks: str, add_mask: bool, var: float = 0.0):
        super(MaskModelWithNoNoise, self).__init__()

        self.mean_system_response = MaskMeanSystemResponse(path_masks, add_mask)
        self.noise_covariance = ScalarNoiseCovariance(var=var)


    def fix_state(self, x_0, fabric, eval):

        # reset the random component used before fix_state call
        self.mean_system_response.reset_random_state(x_0, fabric, eval)


    def pinvA(self, y):
        return self.mean_system_response.pinvA(y)
    

    def pinvA_A(self, x):
        return self.mean_system_response.pinvA(self.mean_system_response.A(x))
    

    def pinvA_Sigma_sqrt(self, x):
        return self.mean_system_response.pinvA(self.noise_covariance.sqrtSigma(x))