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
    
    # Data path Parameters
    parser.add_argument("--data_dir", type=str, default="data/v2", help="Path to the data directory")
    parser.add_argument("--data_split", type=str, default="object_unseen", choices=["object_unseen", "scene_unseen"], help="Evaluation setting to use")

    # Mode Parameters
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode to run the script")
    parser.add_argument("--validate_after_n_epochs", type=int, default=1, help="Validate the model after every n epochs")
    
    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")

    # Model and Training Parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--loss_choice", type=str, default="L2", help="Loss function choice")
    
    # Checkpoints Parameters
    parser.add_argument("--resume_training", action="store_true", default=False, help="Resume training from checkpoint")

    return parser.parse_args()
