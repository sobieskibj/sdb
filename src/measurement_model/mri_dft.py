import abc
import torch

from .base import BaseLinearModelWithGaussianNoise

from mean_system_response.mri_dft import MRIDFTMeanSystemResponse
from noise_covariance.scalar import ScalarNoiseCovariance

import logging

log = logging.getLogger(__name__)


class MRIDFTModelWithScalarCovarianceNoise(BaseLinearModelWithGaussianNoise):
    def __init__(
        self,
        path_masks: str,
        var: float,
    ):
        super(MRIDFTModelWithScalarCovarianceNoise, self).__init__()
        self.noise_covariance = ScalarNoiseCovariance(var)
        self.mean_system_response = MRIDFTMeanSystemResponse(path_masks)

    def fix_state(self, x_0, fabric, eval):
        # reset the random component used before fix_state call
        self.mean_system_response.reset_random_state(x_0, fabric, eval)

    def pinvA(self, y):
        return self.mean_system_response.pinvA(y)

    def pinvA_A(self, x):
        return self.mean_system_response.pinvA(self.mean_system_response.A(x))

    def pinvA_Sigma_sqrt(self, x):
        return self.mean_system_response.pinvA(self.noise_covariance.sqrtSigma(x))
