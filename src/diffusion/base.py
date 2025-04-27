import abc
import torch

from network.base import PredictionType


class BaseDiffusion(abc.ABC, torch.nn.Module):
    '''
    Base class for diffusion models. Each inheritant must implement:
    - training_step which defines what happens during a single forward pass in training
    - validation_step which defines what happens during a single forward pass in validation
    '''

    def __init__(self):
        super(BaseDiffusion, self).__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    @abc.abstractmethod
    def _map_network_output_to_x_0(self, x_0, y_hat, prediction_type: PredictionType):
        ...

    @abc.abstractmethod
    def _map_x_0_to_network_target(self, x_0, x_t, prediction_type: PredictionType):
        ...
    
    @abc.abstractmethod
    def training_step(self):
        ...

    @abc.abstractmethod
    def validation_step(self):
        ...