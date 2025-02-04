import argparse

def get_args():
    """
    Defines and parses command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace.
    """
    parser = argparse.ArgumentParser(description="Training script for coordinate regression")
    
    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")

    # Model and Training Parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--loss_choice", type=str, default="L2", choices=["L1", "L2"], help="Loss function choice")
    
    # Attention parameters
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension for attention")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    
    # Encoder parameters
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze the encoder parameters")
    
    # Map parameters
    parser.add_argument("--map_size", type=int, default=500, help="Size of the map (e.g., 500x500)")
    parser.add_argument("--pixels_per_meter", type=int, default=10, help="Pixels per meter for map scaling")

    # Optimizer Parameters
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "adamw", "rmsprop"], help="Optimizer to use")

    # Scheduler Parameters
    parser.add_argument("--scheduler", type=str, default=None, choices=["none", "step_lr", "cosine_annealing", "exponential_lr", "reduce_on_plateau"], help="Learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=5, help="Step size for StepLR scheduler (if used)")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma value for schedulers")
    parser.add_argument("--patience", type=int, default=10, help="Patience value for ReduceOnPlateau scheduler")

    # Device
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to train on")

    return parser.parse_args()
