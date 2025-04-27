"""Ported from https://github.com/Hammour-steak/GOUB/tree/main/codes/models/modules"""
import enum
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import functools

from .unet_2d_goub_utils import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock,
    LinearAttention,
    PreNorm, 
    Residual)
from .base import BaseNetwork, PredictionType, ConditioningTransformationType, ConditioningType


import logging
log = logging.getLogger(__name__)


class UNet2DGOUD(BaseNetwork):

    def __init__(
            self, 
            in_nc, 
            out_nc, 
            nf, 
            depth, 
            condition_transform_type, 
            condition_type, 
            transform_x, 
            inv_transform_x, 
            transform_cond,
            prediction_type):
        super(UNet2DGOUD, self).__init__()

        # save prediction type
        self.prediction_type = prediction_type

        # use the provided transformations or identity if none 
        self.transform_x = transform_x if transform_x is not None else lambda x: x
        self.inv_transform_x = inv_transform_x if inv_transform_x is not None else lambda x: x
        self.transform_cond = transform_cond if transform_cond is not None else lambda x: x

        # save conditioning type
        self.condition_transform_type = condition_transform_type
        self.condition_type = condition_type
        log.info(f"Conditioning type {condition_type} with transformation {condition_transform_type}")

        # network is conditional if and only if condition_type is different from NONE
        self.conditional = ConditioningType[condition_type] != ConditioningType.NONE

        self.depth = depth

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(
            in_nc*2 if ConditioningTransformationType[condition_transform_type] == ConditioningTransformationType.CONCATENATION else in_nc, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i+1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

        self.log_param_count()


    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward(self, xt, cond, time):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)

        x = self.transform_x(xt)

        if self.conditional:
            cond = self.transform_cond(cond)
        
        if ConditioningTransformationType[self.condition_transform_type] == ConditioningTransformationType.CONCATENATION:

            x = xt - cond
            x = torch.cat([x, cond], dim=1)

        elif ConditioningTransformationType[self.condition_transform_type] == ConditioningTransformationType.EMBEDDING:

            ...

        elif ConditioningTransformationType[self.condition_transform_type] == ConditioningTransformationType.NONE:

            x = xt

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time)

        h = []

        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
            
        x = self.inv_transform_x(x)

        return x


