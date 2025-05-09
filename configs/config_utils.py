from omegaconf import OmegaConf
import os

import os
from omegaconf import OmegaConf, DictConfig

def load_config(
    config_path: str = "eai_pers.yaml",
    base_path: str = "configs",
    verbose: bool = True,
) -> DictConfig:
    """
    Load and merge the base 'default.yaml' with an experiment-specific config.

    Args:
        config_path: Name of the experiment config in 'configs/experiment/'.
        base_path: Root directory for configs (contains 'default.yaml' and 'experiment/').
        verbose: If True, print loaded configs.

    Returns:
        OmegaConf.DictConfig: The merged configuration.
    """
    default_file = os.path.join(base_path, "default.yaml")
    exp_file     = os.path.join(base_path, "experiments", config_path)

    if not os.path.isfile(default_file):
        raise FileNotFoundError(f"Default config not found: {default_file}")
    default_cfg = OmegaConf.load(default_file)

    exp_cfg = OmegaConf.load(exp_file) if os.path.isfile(exp_file) else OmegaConf.create()

    if verbose:
        print(f"Loaded default config from '{default_file}'")
        print(OmegaConf.to_yaml(default_cfg, resolve=True))
        if exp_cfg:
            print(f"\nLoaded experiment config from '{exp_file}'")
            print(OmegaConf.to_yaml(exp_cfg, resolve=True))

    merged = OmegaConf.merge(default_cfg, exp_cfg)
    return merged



def save_config(config, config_path):
    """
    Save an OmegaConf config to a .yaml file.
    """
    with open(config_path, "w") as f:
        OmegaConf.save(config=config, f=f)
        
        
def override_config(config, overrides: dict):
    """
    Override config values with a dictionary of overrides.
    """
    override_cfg = OmegaConf.create(overrides)
    return OmegaConf.merge(config, override_cfg)


def flatten_config(config, parent_key='', sep='.'):
    """
    Flatten a nested config into a single dictionary with dot-separated keys.
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or isinstance(v, OmegaConf):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)