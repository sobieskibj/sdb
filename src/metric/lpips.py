import lpips
import wandb
import torch

from .base import BaseMetric


import logging
log = logging.getLogger(__name__)


class LPIPS(BaseMetric):


    def __init__(self, correct_scale_and_shape: bool):
        super(LPIPS, self).__init__()

        self.net = lpips.LPIPS(net='alex')
        self.correct_scale_and_shape = correct_scale_and_shape
        self.reset()


    def __str__(self):
        return "LPIPS"


    @torch.no_grad()
    def forward(self, x_0, x_0_hat):
        
        # asserts that scale and shape is correct ([0, 1], 3)
        if self.correct_scale_and_shape:
            x_0, x_0_hat = self.map_scale_and_shape(x_0), self.map_scale_and_shape(x_0_hat)

        # map [0, 1] to [-1, 1]
        x_0 = map_01_to_m1p1(x_0)
        x_0_hat = map_01_to_m1p1(x_0_hat)

        # compute lpips and return
        value = self.net(x_0, x_0_hat).sum()

        # increment counter and store value
        self.counter += x_0.shape[0]
        self.value += value


    def compute_and_log(self, fabric, log_prefix=""):

        # gather data from other processes
        counter = fabric.all_reduce(self.counter, reduce_op="sum")
        value = fabric.all_reduce(self.value, reduce_op="sum")

        # log metric only on main process
        if fabric.global_rank == 0:

            # log mean lpips and reset 
            lpips = value / counter
            wandb.log({f"{log_prefix}metrics/lpips": lpips})
            log.info(f"{log_prefix}LPIPS: {lpips}")

        # sync processes since only rank 0 logs all metrics
        fabric.barrier()

        # reset
        self.reset()


    def reset(self):
        self.counter = 0
        self.value = 0.


def map_01_to_m1p1(x):
    return (x - 0.5) * 2