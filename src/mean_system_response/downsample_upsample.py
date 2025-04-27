import torch.nn.functional as F

from .base import BaseMeanSystemResponse


class DownsampleUpsampleMeanSystemResponse(BaseMeanSystemResponse):


    def __init__(self, scale_factor, upsample_A: bool):
        """
        Linear operator representing downsampling by scale_factor and upscaling to initial image size.
        """
        super(DownsampleUpsampleMeanSystemResponse, self).__init__()
        self.scale_factor = scale_factor
        self.upsample_A = upsample_A


    def A(self, x):
        B, C, H, W = x.shape
        x = F.avg_pool2d(x, kernel_size=self.scale_factor, stride=self.scale_factor)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x
    

    def AT(self, y):
        B, C, H, W = y.shape
        y = F.interpolate(y, size=(H * self.scale_factor, W * self.scale_factor), mode='bilinear', align_corners=False)
        return y


    def pinvATA(self, x):
        return (self.scale_factor ** 2) * x


    def reset_random_state(self):
        pass