from omegaconf import OmegaConf
import os

import os
from omegaconf import OmegaConf, DictConfig
# Import the structured DefaultConfig dataclass
from configs.default_structure_config import DefaultConfig

def load_config(
    config_path: str = "eai_pers.yaml",
    base_path: str = "configs",
    verbose: bool = True,
) -> DictConfig:
    """
    Load and merge the schema, base 'default.yaml', and an experiment-specific config.

    Args:
        config_path: Name of the experiment config in 'configs/experiments/'.
        base_path: Root directory for configs (contains 'default.yaml' and 'experiments/').
        verbose: If True, print loaded configs.

    Returns:
        OmegaConf.DictConfig: The merged configuration.
    """
    # Build file paths
    default_file = os.path.join(base_path, "default.yaml")
    exp_file = os.path.join(base_path, "experiments", config_path)

    # Verify default exists
    if not os.path.isfile(default_file):
        raise FileNotFoundError(f"Default config not found: {default_file}")
    # Verify experiment exists
    if not os.path.isfile(exp_file):
        raise FileNotFoundError(f"Experiment config not found: {exp_file}")

    # Create structured schema from dataclass
    schema_cfg = OmegaConf.structured(DefaultConfig)

    # Load raw YAMLs
    default_cfg = OmegaConf.load(default_file)
    exp_cfg = OmegaConf.load(exp_file)

    # Merge: schema -> default -> experiment
    merged_cfg = OmegaConf.merge(schema_cfg, default_cfg, exp_cfg)

    if verbose:
        print("-----------------------------")
        print(f"\nArguments for'{exp_file}'")
        print("-----------------------------")
        print(OmegaConf.to_yaml(merged_cfg, resolve=True))

    return merged_cfg

def save_config(config: DictConfig, config_path: str) -> None:
    """
    Save an OmegaConf config to a .yaml file.
    """
    OmegaConf.save(config=config, f=config_path)

def override_config(config: DictConfig, overrides: dict) -> DictConfig:
    """
    Override config values with a dictionary of overrides.
    """
    override_cfg = OmegaConf.create(overrides)
    return OmegaConf.merge(config, override_cfg)

def flatten_config(config: DictConfig, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten a nested config into a single dictionary with dot-separated keys.
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig) or isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)