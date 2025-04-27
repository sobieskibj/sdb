import wandb
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance


from .base import BaseMetric


import logging

log = logging.getLogger(__name__)


class FID(BaseMetric):
    def __init__(self, dims: int, correct_scale_and_shape: bool, *args, **kwargs):
        super(FID, self).__init__()

        self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]])
        self.correct_scale_and_shape = True  # hardcoded as we use it everywhere
        self.reset()

    def __str__(self):
        return "FID"

    @torch.no_grad()
    def forward(self, x_0, x_0_hat):
        # asserts that scale and shape is correct ([0, 1], 3)
        if self.correct_scale_and_shape:
            x_0, x_0_hat = (
                self.map_scale_and_shape(x_0),
                self.map_scale_and_shape(x_0_hat),
            )

        # obtain features for both batches
        d1_features_ = self.model(x_0)[0]
        d2_features_ = self.model(x_0_hat)[0]

        # and store them
        self.d1_features.append(d1_features_)
        self.d2_features.append(d2_features_)

    def compute_and_log(self, fabric, log_prefix=""):
        # gather data from other processes if any
        if fabric.world_size > 1:
            # remove last element if its shape doesnt agree with others
            same_shapes = self.d1_features[-1].shape[0] == self.d1_features[-2].shape[0]

            if not same_shapes:
                self.d1_features = self.d1_features[:-1]
                self.d2_features = self.d2_features[:-1]

            d1_features = fabric.all_gather(torch.cat(self.d1_features)).flatten(0, 1)
            d2_features = fabric.all_gather(torch.cat(self.d2_features)).flatten(0, 1)

        else:
            d1_features = torch.cat(self.d1_features)
            d2_features = torch.cat(self.d2_features)

        # log metric only on main process
        if fabric.global_rank == 0:
            # concatenate features and convert to numpy
            d1_features = d1_features.numpy(force=True)
            d2_features = d2_features.numpy(force=True)

            # compute mean and sigma for both distributions
            mu_1, sigma_1 = self.get_mu_sigma(d1_features)
            mu_2, sigma_2 = self.get_mu_sigma(d2_features)

            # get fid
            fid_value = calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)

            # log fid
            wandb.log({f"{log_prefix}metrics/fid": fid_value})
            log.info(f"{log_prefix}FID: {fid_value}")

        # sync processes since only rank 0 logs all metrics
        fabric.barrier()

        # reset
        self.reset()

    def get_mu_sigma(self, features):
        mu = np.mean(features.squeeze(), axis=0)
        sigma = np.cov(features.squeeze(), rowvar=False)
        return mu, sigma

    def reset(self):
        self.d1_features = []
        self.d2_features = []


def map_01_to_m1p1(x):
    return (x - 0.5) * 2


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]


class RepeatChannels(torch.nn.Module):
    def __init__(self, repeat=3):
        super().__init__()
        self.repeat = repeat
        # self.dummy_param = torch.nn.Parameter(torch.randn(1))

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Tensor shape: (B, C, H, W)
            C = img.shape[1]
            return img.repeat(1, self.repeat, 1, 1) if C == 1 else img
        else:
            raise TypeError("Input should be a torch.Tensor")


class ScaleToM1P1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.dummy_param = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        x_min = expand(x.flatten(start_dim=1).min(dim=1)[0], x)
        x_max = expand(x.flatten(start_dim=1).max(dim=1)[0], x)
        return 2 * (x - x_min) / (x_max - x_min) - 1


class InterpolateToInception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.dummy_param = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
