from omegaconf import OmegaConf
import os

def load_config(config_path="default.yaml", base_path="configs", verbose=True):
    """
    Load and optionally merge a base default.yaml with the specified config.
    
    Args:
        config_path (str): YAML config to load (e.g., "experiment1.yaml").
        base_path (str): Path to the config directory.
        verbose (bool): Whether to print the loaded config.

    Returns:
        OmegaConf.DictConfig: Merged configuration object.
    """
    conf_path = os.path.join(base_path, config_path)

    # Load base default config
    default_cfg = OmegaConf.load(conf_path) if os.path.exists(conf_path) else OmegaConf.create()

    # Merge them (experiment overrides default)
    config = OmegaConf.merge(default_cfg)

    if verbose:
        print(f"\nLoaded config from '{config}'...\n")
        print(OmegaConf.to_yaml(config, resolve=True))

    return config


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