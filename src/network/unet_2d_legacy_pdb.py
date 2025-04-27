import monai
import torch
from diffusers import UNet2DModel

from .base import BaseNetwork, PredictionType

class UNet2DLegacyPDB(BaseNetwork):
    '''
    TODO: we can consider refactoring it / using some existing architecture from papers like I2SB.
    e.g. I think that UNet2DModel from diffusers already comes with time conditioning
    '''
    
    prediction_type = PredictionType.X_0

    def __init__(self,  
                 input_channels: int,
                 time_encoder_hidden_size: int,
                 image_size: int,
                 unet_in_channels: int, 
                 unet_base_channels: int,
                 unet_out_channels: int,
                 conditional_channels: int):

        super().__init__()

        self.conditional_channels = conditional_channels

        # Create a UNet2DModel for noise prediction given x_t and t
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=unet_out_channels,
            layers_per_block=2,
            norm_num_groups=1,
            block_out_channels=(unet_base_channels, unet_base_channels, 2*unet_base_channels, 2*unet_base_channels, 4*unet_base_channels, 4*unet_base_channels),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        self.time_encoder = monai.networks.nets.FullyConnectedNet(
            in_channels=1, 
            out_channels=unet_in_channels - input_channels - conditional_channels, 
            hidden_channels=[time_encoder_hidden_size, time_encoder_hidden_size, time_encoder_hidden_size],
            dropout=None, 
            act='PRELU', 
            bias=True, 
            adn_ordering='NDA')

    def _forward(self, z_t, t):
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)
        
        # apply the MLP to the status concatenated with the time
        t_enc = self.time_encoder(t)

        # now repeat it so it's the same size as z_t
        t_enc = t_enc.unsqueeze(-1).unsqueeze(-1)
        t_enc = t_enc.repeat(1, 1, z_t.shape[2], z_t.shape[3])

        # now concatenate it with z_t
        z_t_and_t = torch.cat([z_t, t_enc], dim=1)

        # now apply the UNet
        residual = self.unet(z_t_and_t, t.squeeze())[0]

        z_0_hat = z_t + (torch.sqrt(t)) * residual

        return z_0_hat

    def _forward_conditional(self, z_t, x_tilde, t):
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)

        # apply the MLP to the status concatenated with the time
        t_enc = self.time_encoder(t)

        # now repeat it so it's the same size as z_t
        t_enc = t_enc.unsqueeze(-1).unsqueeze(-1)
        t_enc = t_enc.repeat(1, 1, z_t.shape[2], z_t.shape[3])

        # concatenate z_t, t_enc, and x_tilde
        z_t_and_t_and_x_tilde = torch.cat([z_t, t_enc, x_tilde], dim=1)

        # apply the UNet
        residual = self.unet(z_t_and_t_and_x_tilde, t.squeeze())[0]

        z_0_hat = z_t + (torch.sqrt(t)) * residual

        return z_0_hat

    def forward(self, z_t, t, conditional=None):
        if conditional is None:
            return self._forward(z_t, t)
        else:
            # re-interpret, t is actually x_tilde
            x_tilde = t.clone()
            t = conditional
            return self._forward_conditional(z_t, x_tilde, t)