"""Ported from https://github.com/Hammour-steak/GOUB/blob/main/codes/utils/sde_utils.py"""

import abc
from torch import nn
from tqdm import tqdm


class BaseGOUB(abc.ABC, nn.Module):
    def __init__(self, T, device=None):
        super(BaseGOUB, self).__init__()
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):  # train process
        return x - self.sde_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)
        return x


class BaseIRSDE(abc.ABC):
    def __init__(self, T, device=None):
        super(BaseIRSDE, self).__init__
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


import torch

from network.base import PredictionType


class BaseLinearSDE(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def sample_x_t_minus_dt_given_x_t(self, x, t):
        pass

    @abc.abstractmethod
    def sample_x_t_given_x_0(self, x, t):
        pass

    @abc.abstractmethod
    def _map_network_output_to_reverse_drift(
        self, x_t, output, t, dt, prediction_type, is_last
    ):
        pass

    def _map_network_output_to_x_0(
        self, x_t, output, t, prediction_type: PredictionType
    ):
        if PredictionType[prediction_type] == PredictionType.MEAN:
            x_0_hat = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            x_0_hat = output

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            x_0_hat = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")

        return x_0_hat

    def _map_x_0_to_network_target(self, x_0, x_t, t, prediction_type):
        if PredictionType[prediction_type] == PredictionType.MEAN:
            target = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            target = x_0

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            target = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")

        return target


class BaseScalarSDE(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def sample_x_t_minus_dt_given_x_t(self, x, t):
        pass

    @abc.abstractmethod
    def sample_x_t_given_x_0(self, x, t):
        pass

    @abc.abstractmethod
    def _map_network_output_to_reverse_drift(
        self, x_t, output, t, dt, prediction_type, is_last
    ):
        pass

    def _map_network_output_to_x_0(
        self, x_t, output, t, prediction_type: PredictionType
    ):
        if PredictionType[prediction_type] == PredictionType.MEAN:
            x_0_hat = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            x_0_hat = output

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            x_0_hat = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")

        return x_0_hat

    def _map_x_0_to_network_target(self, x_0, x_t, t, prediction_type):
        if PredictionType[prediction_type] == PredictionType.MEAN:
            target = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            target = x_0

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            target = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")

        return target


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]
