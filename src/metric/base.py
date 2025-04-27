import cv2
import abc
import math
import torch
import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF


class BaseMetric(abc.ABC, torch.nn.Module):
    """Base class for metrics."""

    def __init__(self):
        super(BaseMetric, self).__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)


    @abc.abstractmethod
    def compute_and_log(self):
        """Compute metric from intermediate states and log results."""
        pass


    def map_scale_and_shape(self, x):
        # get shapes
        B, C, H, W = x.shape

        # scale to [0, 1] range
        x = min_max_scale(x)

        # map to uint8 and then back to float32
        x = TF.convert_image_dtype(x, torch.uint8) / 255.

        # repeat channels to have 3
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        
        return x


    def bgr2ycbcr(self, img, only_y=True):
        '''bgr version of rgb2ycbcr
        only_y: only return Y channel
        Input:
            uint8, [0, 255]
            float, [0, 1]
        '''
        in_img_type = img.dtype
        img.astype(np.float32)
        if in_img_type != np.uint8:
            img *= 255.
        # convert
        if only_y:
            rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        else:
            rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                                [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
        if in_img_type == np.uint8:
            rlt = rlt.round()
        else:
            rlt /= 255.
        return rlt.astype(in_img_type)
    

    def tensor2img(self, tensor, out_type=np.uint8, min_max=(0, 1)):
        """
        Converts a torch Tensor into an image Numpy array
        Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
        """
        tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        n_dim = tensor.dim()
        if n_dim == 4:
            n_img = len(tensor)
            img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        elif n_dim == 3:
            img_np = tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        elif n_dim == 2:
            img_np = tensor.numpy()
        else:
            raise TypeError(
                "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                    n_dim
                )
            )
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
            # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
        return img_np.astype(out_type)


    def calculate_psnr(self, img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * math.log10(255.0 / math.sqrt(mse))


    def calculate_ssim(self, img1, img2):
        """calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        """
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def min_max_scale(tensor):
    B = tensor.shape[0]
    tensor = tensor - tensor.flatten(start_dim=1).min(1)[0].view(B, 1, 1, 1)
    tensor = tensor / tensor.flatten(start_dim=1).max(1)[0].view(B, 1, 1, 1)
    return tensor