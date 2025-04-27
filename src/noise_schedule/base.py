import abc
import torch


class BaseNoiseSchedule(abc.ABC, torch.nn.Module):

    def __init__(self):
        super(BaseNoiseSchedule, self).__init__()