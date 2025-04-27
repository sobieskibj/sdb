import abc
import torch
import enum


import logging
log = logging.getLogger(__name__)


class PredictionType(enum.Enum):
    MEAN = "MEAN"
    X_0 = "X_0"
    EPSILON = "EPSILON"
    SCORE = "SCORE"
    

class ConditioningTransformationType(enum.Enum):
    NONE = "NONE"
    CONCATENATION = "CONCATENATION"
    EMBEDDING = "EMBEDDING"


class ConditioningType(enum.Enum):
    NONE = "NONE"
    PSEUDOINVERSE_RECONSTRUCTION = "PSEUDOINVERSE_RECONSTRUCTION"
    MEASUREMENT = "MEASUREMENT"


class BaseNetwork(torch.nn.Module, abc.ABC):


    def __init__(self):
        super(BaseNetwork, self).__init__()


    def log_param_count(self):
        total_params = 0
        for name, param in self.named_parameters():
            param_count = param.numel()
            total_params += param_count
        log.info(f"Total number of parameters: {total_params}")