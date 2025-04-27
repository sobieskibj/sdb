import wandb
import torch
from tqdm import tqdm
from pathlib import Path
from ema_pytorch import EMA
from omegaconf import DictConfig
from hydra.utils import instantiate
from collections import OrderedDict

import utils

import logging

log = logging.getLogger(__name__)


def get_fabric(config):
    """Instantiate Fabric object, set seed and launch."""
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_components(config, fabric):
    """
    Instantiate all torch objects with their optional optimizers and setup with Fabric.
    To avoid problems with distributed strategies when parallelizing objects with no parameters
    to train, we additionally set .requires_grad_(True) on each of them.
    """

    # init network and its optimizer
    network = instantiate(config.network)
    ema_network = fabric.setup(
        instantiate(config.ema_network)(model=network).requires_grad_(True)
    )

    # init diffusion
    diffusion = fabric.setup(instantiate(config.diffusion))

    # init metrics
    metrics = [fabric.setup(instantiate(m)) for m in config.metric.values()]

    return network, ema_network, diffusion, metrics


def get_val_dataloader(config, fabric):
    """Instantiate dataloader and setup with Fabric."""
    return fabric.setup_dataloaders(instantiate(config.val_dataloader))


def run_validation(config, fabric, dataloader, diffusion, network, metrics, log_prefix):
    # network eval state
    network.eval()

    # get loop length
    total_length = len(dataloader.dataset) // dataloader.batch_size + 1

    # iterate over validation dataloader
    for batch_idx, x_0 in enumerate(tqdm(dataloader, total=total_length)):
        # sample from diffusion
        x_0_hat = diffusion.validation_step(
            fabric, batch_idx, x_0, network, config.exp.log_assets, log_prefix
        )

        # accumulate data for each metric
        for metric in metrics:
            if batch_idx == 0:
                log.info(f"Accumulating {metric.module}")
            metric(x_0, x_0_hat)

    # post validation, compute all metrics and make logs
    for metric in metrics:
        log.info(f"Computing {metric.module}")

        # pass fabric to compute metrics on rank 0
        metric.compute_and_log(fabric, log_prefix)

        # sync processes since only rank 0 logs all metrics
        fabric.barrier()

    # reset any validation-dependent state
    diffusion.measurement_likelihood.fix_state(x_0, fabric, eval=False)


def get_paths_checkpoints(config, fabric):
    # path to ckpts dir
    path = Path(config.exp.train_log_dir) / "checkpoints"

    # find all checkpoints and get sorted truncated list
    paths = list(path.glob("*.ckpt"))
    paths = sorted(paths, key=lambda x: int(x.stem.split("_")[1]))
    paths = paths[
        config.exp.n_ckpts_to_skip : config.exp.n_ckpts_total : config.exp.every_n_ckpt
    ]
    log.info(f"Evaluating {len(paths)} checkpoints")

    # return dictionary with key indicating iteration number
    return OrderedDict([(int(v.stem.split("_")[1]), v) for v in paths])


def load_checkpoint(fabric, path_checkpoint, network, ema_network):
    # load state dict from path
    state_dict = fabric.load(path_checkpoint)

    # set state for each module
    if all([e in state_dict.keys() for e in ["network", "ema_network"]]):
        # in this case we are loading pdb
        network.load_state_dict(state_dict["network"])
        ema_network.ema_model.load_state_dict(state_dict["ema_network"])
        ema_network.step = torch.tensor(state_dict["iteration"])
        eval_ema = True

    else:
        # in other cases we are loading a different method
        network.load_state_dict(state_dict)
        # ema is not evaluated for other methods as we use only provided checkpoints
        eval_ema = False

    return eval_ema


def run(config: DictConfig):
    torch.multiprocessing.set_start_method("spawn")
    utils.hydra.preprocess_config(config)
    config = utils.hydra.combine_configs(
        config,
        Path(config.exp.train_log_dir).parents[1],
        key="train_config",
        override_exceptions=["log_dir"],
    )
    utils.wandb.setup_wandb(config)

    log.info("Launching Fabric")
    fabric = get_fabric(config)

    # context manager to automatically move newly created tensors to correct device
    with fabric.init_tensor():
        log.info("Initializing components")
        network, ema_network, diffusion, metrics = get_components(
            config.train_config, fabric
        )

        log.info("Initializing dataloaders")
        val_dataloader = get_val_dataloader(config, fabric)

        log.info("Searching for checkpoints")
        paths_checkpoints = get_paths_checkpoints(config, fabric)

        log.info("Starting validation")

        # iterates over checkpoint files and evaluates performance
        for checkpoint_id, path_checkpoint in paths_checkpoints.items():
            # loads states from the provided checkpoint path
            log.info(f"Evaluating checkpoint from iteration {checkpoint_id}")
            eval_ema = load_checkpoint(fabric, path_checkpoint, network, ema_network)

            # validate default network
            log.info("Validating default network")
            run_validation(
                config, fabric, val_dataloader, diffusion, network, metrics, ""
            )

            # validate ema network
            if eval_ema and config.exp.eval_ema:
                log.info("Validating EMA network")
                run_validation(
                    config,
                    fabric,
                    val_dataloader,
                    diffusion,
                    ema_network.ema_model,
                    metrics,
                    "ema_",
                )
