import torch

from .base import BaseNetwork, ConditioningType, PredictionType


import logging
log = logging.getLogger(__name__)


class DebugNetwork(BaseNetwork):

    
    def __init__(self):
        super(DebugNetwork, self).__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.condition_type = "PSEUDOINVERSE_RECONSTRUCTION"
        self.prediction_type = "X_0"
    
    def forward(self, xt, cond, time):
        return xt * self.dummy_parameter


    def prediction_type(self):
        pass