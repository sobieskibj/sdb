from .base import BaseNoiseCovariance

class ScalarNoiseCovariance(BaseNoiseCovariance):

    def __init__(self, var: float):
        super(ScalarNoiseCovariance, self).__init__()
        self.var = var

    def Sigma(self, x):
        return self.var * x
    
    def sqrtSigma(self, x):
        return self.var ** (1 / 2) * x
    
    def invSigma(self, x):
        return x / self.var