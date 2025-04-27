import abc
import torch

from .base import BaseLinearModelWithGaussianNoise

from mean_system_response.ct_projector import CTProjectorMeanSystemResponse
from noise_covariance.scalar import ScalarNoiseCovariance

import logging
log = logging.getLogger(__name__)

class CTProjectorModelWithScalarCovarianceNoise(BaseLinearModelWithGaussianNoise):
    """
    NOTE: comment on details
    """

    def __init__(self, path_assets: str, var: float, min_snr: float, direct: bool, use_idx_null: bool):
        super(CTProjectorModelWithScalarCovarianceNoise, self).__init__()
        self.noise_covariance = ScalarNoiseCovariance(var)
        self.mean_system_response = CTProjectorMeanSystemResponse(
            path_assets, min_singular_value=self.get_min_singular_value(var, min_snr, direct), use_idx_null=use_idx_null)


    def get_min_singular_value(self, var, min_snr, direct):
        """
        Provides minimum singular value based on noise std and minimum SNR.
        """
        if direct:
            # min_snr is treated as minimum singular value
            return min_snr
        else:
            std = var ** (1 / 2)
            return min_snr * std


    def fix_state(self, *args, **kwargs):
        pass


    def pinvA(self, y):
        return self.mean_system_response.pinvA(y)
    

    def pinvA_A(self, x):
        return self.mean_system_response.pinvA(self.mean_system_response.A(x))
    

    def pinvA_Sigma_sqrt(self, x):
        return self.mean_system_response.pinvA(self.noise_covariance.sqrtSigma(x))