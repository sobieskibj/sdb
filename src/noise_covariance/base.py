import abc
import torch


class BaseNoiseCovariance(abc.ABC, torch.nn.Module):
    '''Base class for noise covariance instances.'''

    def Sigma(self, x):
        '''Returns \Sigma @ x.'''
        ...

    def sqrt(self, x):
        '''Returns \Sigma^{\frac{1}{2}} @ x.'''
        ...

    def invSigma(self, x):
        '''Returns \Sigma^{-1} @ x.'''
        ...
    
    def forward(self, x):
        return self.Sigma(x)