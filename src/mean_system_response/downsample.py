import torch
import torch.nn.functional as F

from .base import BaseMeanSystemResponse


class DownsampleMeanSystemResponse(BaseMeanSystemResponse):


    def __init__(self, scale_factor):
        """
        Linear operator representing downsampling by scale_factor and upscaling to initial image size.
        """
        super(DownsampleMeanSystemResponse, self).__init__()
        self.scale_factor = scale_factor


    def A(self, x):
        return F.avg_pool2d(x, kernel_size=self.scale_factor, stride=self.scale_factor)


    def AT(self, y):
        B, C, H, W = y.shape
        y = F.interpolate(y, size=(H * self.scale_factor, W * self.scale_factor), mode='nearest')
        return y / (self.scale_factor ** 2)
    
    
    def pinvATA(self, x):
        return self.scale_factor ** 2 * x


    def reset_random_state(self):
        pass


def min_max_scale(tensor):
    B = tensor.shape[0]
    tensor = tensor - tensor.flatten(start_dim=1).min(1)[0].view(B, 1, 1, 1)
    tensor = tensor / tensor.flatten(start_dim=1).max(1)[0].view(B, 1, 1, 1)
    return tensor