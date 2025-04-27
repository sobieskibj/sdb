import abc
import torch


class BaseLinearModelWithGaussianNoise(abc.ABC, torch.nn.Module):
    '''
    Base class for measurement models of noisy linear inverse problems of the form

    y = Ax + \Sigma^{\frac{1]{2}} \epsilon,

    where:
        x is the ground truth
        A is the mean system response matrix
        \epsilon is the standard Gaussian noise
        \Sigma is a positive semi-definite matrix representing noise covariance
        y is the resulting measurement
    
    '''


    def __init__(self):
        super(BaseLinearModelWithGaussianNoise, self).__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)


    def sample(self, x):
        '''Given x, returns y = Ax + \Sigma^{\frac{1]{2}} \epsilon'''
        Ax = self.mean_system_response(x)
        epsilon = torch.randn_like(Ax)
        corr_epsilon = self.noise_covariance.sqrtSigma(epsilon)
        return Ax + corr_epsilon
    

    def forward(self, x):
        return self.sample(x)
    

    @abc.abstractmethod
    def fix_state(self):
        ...