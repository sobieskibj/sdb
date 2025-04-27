import wandb
import torch
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms.functional as TF

from .base import BaseMetric


import logging
log = logging.getLogger(__name__)


class PSNRSSIM(BaseMetric):


    def __init__(self, correct_scale_and_shape: bool):
        super(PSNRSSIM, self).__init__()
        
        self.correct_scale_and_shape = correct_scale_and_shape
        self.reset()


    def __str__(self):
        return "PSNRSSIM"


    @torch.no_grad()
    def forward(self, x_0, x_0_hat):
        """Ported from https://github.com/Hammour-steak/GOUB/blob/main/codes/utils/img_utils.py#L216"""

        # asserts that scale and shape is correct ([0, 1], 3)
        if self.correct_scale_and_shape:
            x_0, x_0_hat = self.map_scale_and_shape(x_0), self.map_scale_and_shape(x_0_hat)

        ####
        test_results = OrderedDict()
        test_results["psnr"] = []
        test_results["ssim"] = []
        test_results["psnr_y"] = []
        test_results["ssim_y"] = []
        ####
        
        for output, GT_ in zip(x_0, x_0_hat):
            ####
            output = self.tensor2img(output.squeeze())  # uint8
            GT_ = self.tensor2img(GT_.squeeze())  # uint8

            gt_img = GT_ / 255.0
            sr_img = output / 255.0

            cropped_sr_img = sr_img
            cropped_gt_img = gt_img

            psnr = self.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = self.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)

            if len(gt_img.shape) == 3:
                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = self.bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = self.bgr2ycbcr(gt_img, only_y=True)
   
                    cropped_sr_img_y = sr_img_y
                    cropped_gt_img_y = gt_img_y
   
                    psnr_y = self.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    ssim_y = self.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )

                    test_results["psnr_y"].append(psnr_y)
                    test_results["ssim_y"].append(ssim_y)

            else:
                test_results["psnr_y"].append(psnr)
                test_results["ssim_y"].append(ssim)
            ####
        
        self.psnr_y += test_results["psnr_y"]
        self.psnr += test_results["psnr"]
        self.ssim_y += test_results["ssim_y"]
        self.ssim += test_results["ssim"]


    def compute_and_log(self, fabric, log_prefix=""):

        if fabric.world_size > 1:
            # gather data from other processes if there are any
            psnr_y = torch.cat(fabric.all_gather(self.psnr_y)).flatten()
            psnr = torch.cat(fabric.all_gather(self.psnr)).flatten()
            ssim_y = torch.cat(fabric.all_gather(self.ssim_y)).flatten()
            ssim = torch.cat(fabric.all_gather(self.ssim)).flatten()
        
        else:
            # otherwise just extract the list
            psnr_y = self.psnr_y
            psnr = fabric.all_gather(self.psnr)
            ssim_y = fabric.all_gather(self.ssim_y)
            ssim = fabric.all_gather(self.ssim)

        # log metric only on main process
        if fabric.global_rank == 0:

            psnr_y = sum(psnr_y) / len(psnr_y)
            psnr = sum(psnr) / len(psnr)
            ssim_y = sum(ssim_y) / len(ssim_y)
            ssim = sum(ssim) / len(ssim)

            # log mean psnr and ssim
            wandb.log({
                f"{log_prefix}metrics/psnr_y": psnr_y,
                f"{log_prefix}metrics/psnr": psnr,
                f"{log_prefix}metrics/ssim_y": ssim_y,
                f"{log_prefix}metrics/ssim": ssim})

            log.info(f"{log_prefix}PSNR: {psnr}")
            log.info(f"{log_prefix}SSIM: {ssim}")
            log.info(f"{log_prefix}PSNRy: {psnr_y}")
            log.info(f"{log_prefix}SSIMy: {ssim_y}")

        # sync processes since only rank 0 logs all metrics
        fabric.barrier()

        # reset containers
        self.reset()


    def reset(self):
        self.psnr_y = []
        self.psnr = []
        self.ssim_y = []
        self.ssim = []