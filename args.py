import argparse

def get_args():
    """
    Defines and parses command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace.
    """
    parser = argparse.ArgumentParser(description="Training & Eval script for Eai-Pers repo")
    
    # Configuration file
    parser.add_argument("--config", type=str, default="eai_pers.yaml", help="Path to the configuration file")
    
    # Checkpoints Parameters
    parser.add_argument("--resume_training", action="store_true", default=False, help="Resume training from checkpoint")

    return parser.parse_args()
