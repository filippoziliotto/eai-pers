import argparse

def get_args():
    """
    Defines and parses command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace.
    """
    parser = argparse.ArgumentParser(description="Training script for coordinate regression")
    
    # Data path Parameters
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--data_split", type=str, default="object_unseen", choices=["object_unseen", "scene_unseen"], help="Evaluation setting to use")

    # Mode Parameters
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode to run the script")
    parser.add_argument("--validate_after_n_epochs", type=int, default=1, help="Validate the model after every n epochs")
    
    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    
    # Augmentations Parameters
    parser.add_argument("--use_aug", action="store_true", default=False, help="Use augmentations for training")
    parser.add_argument("--aug_prob", type=float, default=0.5, help="Probability of applying augmentations")
    parser.add_argument("--use_horizontal_flip", action="store_true", default=False, help="Use horizontal flip")
    parser.add_argument("--use_vertical_flip", action="store_true", default=False, help="Use vertical flip")
    parser.add_argument("--use_random_crop", action="store_true", default=False, help="Use rotation")
    parser.add_argument("--use_random_rotate", action="store_true", default=False, help="Use random crop")
    parser.add_argument("--use_desc_aug", action="store_true", default=False, help="Use description augmentation")
    
    # Model and Training Parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--loss_choice", type=str, default="L2", help="Loss function choice")
    parser.add_argument("--loss_scaling", type=float, default=0.3, help="Scaling factor for hybrid loss ( scale * l1 + (1-scale) * l2 )")
    
    # Checkpoints Parameters
    parser.add_argument("--save_checkpoint", action="store_true", default=False, help="Save model checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="model/checkpoints/model.pth", help="Path to save model checkpoints")
    parser.add_argument("--load_checkpoint", action="store_true", default=False, help="Load model checkpoints")
    
    # Attention parameters
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension for attention")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    
    # Encoder parameters
    parser.add_argument("--freeze_encoder", action="store_true", default=True, help="Freeze the encoder parameters")
    
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
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to train on")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb for logging")
    parser.add_argument("--run_name", type=str, default="none", help="Run name for wandb")

    return parser.parse_args()
