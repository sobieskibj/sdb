import wandb
import torch
from tqdm import tqdm
from pathlib import Path
from ema_pytorch import EMA
from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.fabric.utilities import AttributeDict

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
    optimizer = instantiate(config.optimizer)(params=network.parameters())
    lr_scheduler = instantiate(config.lr_scheduler)(optimizer=optimizer)
    network, optimizer = fabric.setup(network, optimizer)
    ema_network = fabric.setup(
        instantiate(config.ema_network)(model=network).requires_grad_(True)
    )
    wandb.watch([network, ema_network], log_freq=64)

    # init diffusion
    diffusion = fabric.setup(instantiate(config.diffusion))

    # init loss function
    loss_fn = instantiate(config.loss_function)

    # init metrics
    metrics = [fabric.setup(instantiate(m)) for m in config.metric.values()]

    # optionally load state from checkpoint
    if config.exp.load_ckpt is not None:
        # get state dict
        state_dict = fabric.load(config.exp.load_ckpt)

        # load state with each module
        network.load_state_dict(state_dict["network"])
        ema_network.ema_model.load_state_dict(state_dict["ema_network"])
        ema_network.step = torch.tensor(state_dict["iteration"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

    return network, ema_network, optimizer, lr_scheduler, diffusion, loss_fn, metrics


def get_train_val_dataloaders(config, fabric):
    """Instantiate dataloaders and setup with Fabric."""
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        instantiate(config.train_dataloader), instantiate(config.val_dataloader)
    )
    return train_dataloader, val_dataloader


def save_state_dict(
    config, fabric, network, ema_network, optimizer, lr_scheduler, iteration, epoch
):
    # create state dict
    state_dict = AttributeDict(
        network=network,
        ema_network=ema_network.ema_model.state_dict(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        iteration=iteration,
        epoch=epoch,
    )

    # optionally create directory for saving and save
    path_wandb = Path(config.exp.log_dir) / "wandb"
    wandb_subdir = [p for p in path_wandb.iterdir() if wandb.run.id in str(p)][0]
    path_save = wandb_subdir / "checkpoints" / f"checkpoint_{iteration}.ckpt"
    path_save.parent.mkdir(exist_ok=True)
    fabric.save(path_save, state_dict)


def get_epoch_offset(config, fabric):
    # if loading from checkpoint, get the number of last training epoch
    if config.exp.load_ckpt is not None:
        offset = fabric.load(config.exp.load_ckpt)["epoch"] + 1
        log.info(f"Resuming from epoch={offset}")

    # otherwise set to 0
    else:
        offset = 0

    return offset


def run_validation(
    config, pbar, fabric, dataloader, diffusion, network, metrics, log_prefix
):
    # network eval state
    network.eval()

    # iterate over validation dataloader
    for batch_idx, x_0 in enumerate(dataloader):
        # update progress bar
        pbar.set_description(f"[{batch_idx=}]")

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

    # reset any validation-dependent state
    diffusion.measurement_likelihood.fix_state(x_0, fabric, eval=False)


def get_iteration(config, fabric, epoch_idx, batch_idx, dataloader):
    full_batch_idx = batch_idx // (config.exp.batch_accum * fabric.world_size)
    n_batches_per_epoch = len(dataloader.dataset) // (
        config.exp.batch_accum * fabric.world_size * config.train_dataloader.batch_size
    )
    prev_epochs = epoch_idx * n_batches_per_epoch
    current_epoch = full_batch_idx + 1
    return prev_epochs + current_epoch


def run(config: DictConfig):
    torch.multiprocessing.set_start_method("spawn")
    utils.hydra.preprocess_config(config)
    utils.wandb.setup_wandb(config)

    # output logging directory path
    log.info(f"config.exp.log_dir={str(config.exp.log_dir)}")

    log.info("Launching Fabric")
    fabric = get_fabric(config)

    # context manager to automatically move newly created tensors to correct device
    with fabric.init_tensor():
        log.info("Initializing components")
        network, ema_network, optimizer, lr_scheduler, diffusion, loss_fn, metrics = (
            get_components(config, fabric)
        )

        log.info("Initializing dataloaders")
        train_dataloader, val_dataloader = get_train_val_dataloaders(config, fabric)

        log.info("Starting training")

        # create progress bar
        offset = get_epoch_offset(config, fabric)
        pbar = tqdm(range(offset, config.exp.n_epochs))

        for epoch_idx in pbar:
            log.info("Training phase")
            wandb.log({"epoch": epoch_idx})

            # network train state
            network.train()

            for batch_idx, x_0 in enumerate(train_dataloader):
                # current iteration number
                train_iteration = get_iteration(
                    config, fabric, epoch_idx, batch_idx, train_dataloader
                )

                # gradient accumulation indicator
                is_accumulating = (
                    False
                    if config.exp.batch_accum == 1
                    else train_iteration % config.exp.batch_accum == 0
                )

                with fabric.no_backward_sync(network, enabled=is_accumulating):
                    # perform training step defined by diffusion
                    loss = diffusion.training_step(
                        fabric, batch_idx, x_0, loss_fn, network, config.exp.log_assets
                    )

                    # backward pass and gradient step
                    fabric.backward(loss)

                # make step only if batch is accumulated
                if not is_accumulating:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema_network.update()

                # update progress bar and log
                if batch_idx % config.exp.log_frequency == 0:
                    pbar.set_description(
                        f"[{epoch_idx=}][{batch_idx=}][{train_iteration=}][loss={round(loss.item(), 7)}][lr={lr_scheduler.get_last_lr()[0]}]"
                    )
                    wandb.log(
                        {
                            "iteration": train_iteration,
                            "loss/train": loss.item(),
                            "optimizer/lr": lr_scheduler.get_last_lr()[0],
                        }
                    )

            # update learning rate
            lr_scheduler.step(epoch=epoch_idx)

            if (
                epoch_idx % config.exp.validation_frequency == 0
                and config.exp.validation_frequency > 0
            ):
                log.info("Validation phase")

                # validate default network
                log.info("Validating default network")
                run_validation(
                    config,
                    pbar,
                    fabric,
                    val_dataloader,
                    diffusion,
                    network,
                    metrics,
                    "",
                )

                if config.exp.eval_ema:
                    # validate ema network
                    log.info("Validating EMA network")
                    run_validation(
                        config,
                        pbar,
                        fabric,
                        val_dataloader,
                        diffusion,
                        ema_network.ema_model,
                        metrics,
                        "ema_",
                    )

            if epoch_idx % config.exp.save_frequency == 0:
                log.info("Saving checkpoint")

                # save all to fabric checkpoint
                save_state_dict(
                    config,
                    fabric,
                    network,
                    ema_network,
                    optimizer,
                    lr_scheduler,
                    train_iteration,
                    epoch_idx,
                )
