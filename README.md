# [System-embedded Diffusion Bridges](https://arxiv.org/abs/2506.23726)

To run any experiment, this source code uses the following convention:

```
WANDB_MODE={offline/online} HYDRA_FULL_ERROR={0/1} python src/main.py --config-name $CONFIG_NAME
```

`CONFIG_NAME` variable corresponds to the name of the .yaml configuration file from the `configs` directory. To modify parameters from the config, use `path.in.config.to.parameter=<new value>` after `$CONFIG_NAME` following Hydra documentation.

# Resources

Data from the following links should be download and placed in the corresponding directory. This ensures integrity with original implementation.

`data/datasets/brainmri`:

`data/datasets/rsna`:

`data/datasets/div2k`: 

`data/datasets/celebahq`: 

`data/metadata`: 

`data/weights`:
