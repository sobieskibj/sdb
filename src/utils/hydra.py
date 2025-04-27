import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, open_dict

import logging
log = logging.getLogger(__name__)


def preprocess_config(config):
    '''Sets config.exp.log_dir to logging directory and symlinks it to CWD.'''
    
    # get logging directory
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # make symlink in current working directory if its not the same
    date_subdir = log_dir.relative_to(log_dir.parents[1])
    log_cwd = Path.cwd() / "outputs" / date_subdir

    if not log_cwd == log_dir and not log_cwd.exists():
        log_cwd.parent.mkdir(exist_ok=True, parents=True)

        # sometimes, the file already exists and we skip this case here
        try:
            log_cwd.symlink_to(log_dir, target_is_directory=True)

        except FileExistsError:
            log.info("Attempting to symlink to existing directory.")

    # save in config
    config.exp.log_dir = log_dir


def update_interpolations(config, key):
    for key_name, value in config.items():
        if isinstance(value, str) and value.startswith("${"):
            # prepend the specified key to the interpolation
            new_value = value.replace("${", f"${{{key}.")
            config[key_name] = new_value
            log.info(f"Updating {key_name} with {new_value}")
        elif isinstance(value, dict) or OmegaConf.is_dict(value):
            # recursively update interpolations in nested dictionaries
            update_interpolations(value, key)


def override(config_to_override, config_from_which_to_override, override_exceptions):
    # iterate through all keys in config_from_which_to_override
    for key, value in config_from_which_to_override.items():
        if isinstance(value, dict) and key in config_to_override:
            # ff the value is a dictionary and the key exists in config_to_override, recurse into the sub-dictionaries
            override(config_to_override[key], value, override_exceptions)
        else:
            # otherwise, directly override the value in config_to_override with the value from config_from_which_to_override
            if key in config_to_override and config_to_override[key] != value and key not in override_exceptions:
                config_to_override[key] = value
                log.info(f"Overriding {key} with {value}")


def combine_configs(config, run_dir, key, override_exceptions):
    """Combines config with the one in run_dir."""

    # load config from run_dir
    path_run_dir_config = Path(run_dir) / ".hydra" / "config.yaml"
    run_dir_config = OmegaConf.load(path_run_dir_config)

    # dump both to standard dict
    config = OmegaConf.to_container(config, resolve=False)
    run_dir_config = OmegaConf.to_container(run_dir_config, resolve=False)

    # add run dir config under specified key
    config[key] = run_dir_config

    # update interpolations with the new key
    update_interpolations(config, key)

    # override with values from config that overlap
    override(config[key], config, override_exceptions)

    # then go back to omegaconf
    config = OmegaConf.create(config)

    return config
    