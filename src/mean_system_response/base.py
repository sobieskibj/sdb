import abc
import torch


class BaseMeanSystemResponse(abc.ABC, torch.nn.Module):
    '''
    Base class for mean system response instances.
    '''

    def __init__(self):
        super(BaseMeanSystemResponse, self).__init__()

    @abc.abstractmethod
    def reset_random_state(self):
        '''
        If A has some stochastic component, e.g, the mask is chosen randomly, this function
        fixes the randomness to be kept the same until the next call.
        '''
        ...

    @abc.abstractmethod
    def A(self, x):
        '''Returns A @ x.'''
        ...

    @abc.abstractmethod
    def AT(self, x):
        '''Returns A^T @ x.'''
        ...

    @abc.abstractmethod
    def pinvATA(self, x):
        '''Returns (A^T @ A)^+ @ x, where A^+ is the pseudoinverse of A.'''
        ...

    def pinvA(self, x):
        '''Returns A^+ @ x.'''
        return self.pinvATA(self.AT(x))
    
        
    def forward(self, x):
        return self.A(x)