

from .base import BaseNoiseCovariance

class NullSpaceCovariance(BaseNoiseCovariance):
    '''
    Represents the null space noise covariance, i.e., \sigma_{null} (I - A^+ A),
    where \sigma_{null} is implemented as nullspace_noise_variance.
    '''    

    def __init__(self, nullspace_noise_variance, measurement_likelihood):
        
        super(NullSpaceCovariance, self).__init__()

        self.nullspace_noise_variance = nullspace_noise_variance
        self.measurement_likelihood = measurement_likelihood
    

    def _project_to_range_space(self, x):
        '''Maps x to the range space, i.e., A^+ A x.'''
        x = self.measurement_likelihood.mean_system_response.A(x)
        x = self.measurement_likelihood.mean_system_response.pinvA(x)
        return x
    

    def Sigma(self, x):
        '''Maps x to \sigma_{null} (I - A^+ A) x.'''
        return self.nullspace_noise_variance * (x - self._project_to_range_space(x))
    

    def sqrtSigma(self, x):
        '''Maps x to \sigma_{null}^{1/2} (I - A^+ A) x. The implementation is correct since
        it can be shown that (I - A^+ A)^{1/2} = I - A^+ A.'''
        return self.nullspace_noise_variance ** (1 / 2) * (x - self._project_to_range_space(x))
    

    def invSigma(self, x):
        '''Maps x to [\sigma_{null} (I - A^+ A)]^{-1} x'''
        # TODO: is it correct?
        return x
    

    def forward(self, x):
        return self.Sigma(x)