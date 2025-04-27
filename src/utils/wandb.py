import torch
import wandb
import omegaconf


def setup_wandb(config):
    """Sets up W&B run based on config."""
    group, name = config.exp.log_dir.parts[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=config.wandb.project,
        dir=config.exp.log_dir,
        group=group,
        name=name,
        config=wandb_config,
        sync_tensorboard=True,
    )


def min_max_scale(tensor):
    if tensor.shape[1] == 2:
        # for complex-valued representation, log absolute value
        tensor = torch.sqrt(tensor[:, 0] ** 2 + tensor[:, 1] ** 2).log1p().unsqueeze(1)
    B = tensor.shape[0]
    tensor = tensor - tensor.flatten(start_dim=1).min(1)[0].view(B, 1, 1, 1)
    tensor = tensor / tensor.flatten(start_dim=1).max(1)[0].view(B, 1, 1, 1)
    return tensor
