
# Python imports
from typing import Union
import numpy as np
import wandb
import random
import os
import sys

# Torch imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# Global variable to store the previous learning rate
_previous_lr = None

"""
General utility functions
"""

def reshape_map(map_tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Reshape input map to (h*w, n) format.
    
    Args:
        map_tensor (Union[torch.Tensor, np.ndarray]): Input map of shape (h,w) or (h,w,n)
        
    Returns:
        torch.Tensor: Reshaped tensor of shape (h*w, n)
    """
    # Convert numpy array to tensor if needed
    if isinstance(map_tensor, np.ndarray):
        map_tensor = torch.from_numpy(map_tensor)
        
    # Ensure input is a tensor
    if not isinstance(map_tensor, torch.Tensor):
        raise TypeError("Input must be either numpy array or torch tensor")
    
    # Add channel dimension if input is 2D
    if map_tensor.dim() != 4:
        raise ValueError(f"Expected 2D or 3D input, got {map_tensor.dim()}D")
    
    # Reshape using view for better memory efficiency
    b, h, w, n = map_tensor.shape
    return map_tensor.view(b, -1, n)

"""
Seed Utils
"""
def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(' ----------------')
    print(f"| Set Seed {seed} |")
    print(' ----------------')
    return

"""
Optimizer Utils
"""

def get_optimizer(optimizer_name, model, lr, weight_decay=0.0, scheduler_name=None, **scheduler_kwargs):
    """
    Returns a PyTorch optimizer (and optionally a scheduler) based on the given names.

    Args:
        optimizer_name (str): Name of the optimizer ('adam', 'sgd', etc.).
        model (torch.nn.Module): The model to optimize.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for regularization (default: 0.0).
        scheduler_name (str, optional): Name of the scheduler ('step_lr', 'cosine_annealing', etc.) or 'none'.
                                        Defaults to None.
        **scheduler_kwargs: Additional keyword arguments for scheduler initialization.
            For example, you might pass `num_epochs=100` or `step_size=30`.

    Returns:
        tuple: (optimizer, scheduler) where scheduler is either an instance of a scheduler or None.
    """
    optimizer_name = optimizer_name.lower()
    
    # Optimizer selection
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}")

    print(f"Optimizer loaded: {optimizer_name}")
    
    # Scheduler selection
    scheduler = None
    if scheduler_name is not None and scheduler_name.lower() != "none":
        scheduler_name_clean = scheduler_name.lower()
        if scheduler_name_clean == "step_lr":
            if 'step_size' not in scheduler_kwargs:
                raise ValueError("step_size must be provided for 'step_lr' scheduler.")
            gamma = scheduler_kwargs.get('gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=scheduler_kwargs['step_size'],
                                                        gamma=gamma)
        elif scheduler_name_clean == "cosine_annealing":
            if 'num_epochs' not in scheduler_kwargs:
                raise ValueError("num_epochs must be provided for 'cosine_annealing' scheduler.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=scheduler_kwargs['num_epochs'])
        elif scheduler_name_clean == "exponential_lr":
            gamma = scheduler_kwargs.get('gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            
        elif scheduler_name_clean == "reduce_on_plateau":
            gamma = scheduler_kwargs.get('gamma', 0.1)
            patience = scheduler_kwargs.get('patience', 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=gamma,
                                                                   patience=patience)
        else:
            raise ValueError(f"Unsupported scheduler name: {scheduler_name_clean}")
        print(f"Scheduler loaded: {scheduler_name_clean}")
    else:
        print("No scheduler loaded.")

    return optimizer, scheduler

"""
Logging & WB Utils
"""
def args_logger(args):
    """
    Logs the arguments to the console.
    
    Args:
        args: The parsed arguments.
    """
    # Sort args by name

    print(' ----------------')
    print("|    Arguments   |")
    print(' ----------------')
    for arg in sorted(vars(args)):
        print(f"| {arg}: {getattr(args, arg)}")
    print(' ----------------')
    return

def log_lr_scheduler(optimizer):
    """
    Log the current learning rate of the optimizer to the console only if it changes.
    
    Args:
        optimizer: The optimizer object.
    """
    global _previous_lr  # Access the global variable defined in this module
    
    # Get the current learning rate from the first parameter group
    current_lr = optimizer.param_groups[0]['lr']
    
    # If _previous_lr is not set, this is the first call
    if _previous_lr is None:
        print(f"Initial learning rate: {current_lr:.0e}")
    # Otherwise, print only if the learning rate has changed
    elif current_lr != _previous_lr:
        print(f"Learning rate changed: {_previous_lr:.0e} -> {current_lr:.0e}")
        
    # Update the previous learning rate for the next call
    _previous_lr = current_lr
    return

def log_epoch_metrics(epoch, optimizer, train_loss, train_acc, val_loss=None, val_acc=None):
    """
    Log training metrics, and optionally validation metrics if provided.

    Args:
        epoch (int): Current epoch.
        optimizer (torch.optim.Optimizer): Optimizer, used here to log the current learning rate.
        train_loss (float): Training loss.
        train_acc (dict): Dictionary of training accuracies.
        val_loss (float, optional): Validation loss.
        val_acc (dict, optional): Dictionary of validation accuracies.
    """
    metrics = {
        "Epoch": epoch+1,
        "Learning Rate": optimizer.param_groups[0]['lr'],
        "Train Loss": train_loss,
    }
    metrics.update({f"Train Acc [{k}]": v for k, v in train_acc.items()})
    
    if val_loss is not None and val_acc is not None:
        metrics["Val Loss"] = val_loss
        metrics.update({f"Val Acc [{k}]": v for k, v in val_acc.items()})
        
    wandb.log(metrics)
    return

def generate_wandb_run_name(args):
    """Generate a concise and descriptive W&B run name based on selected arguments."""
    
    if args.run_name not in 'none':
        return args.run_name
    
    important_args = {
        "batch": args.batch_size,
        "lr": args.lr,
        "wd": args.weight_decay,
        "epochs": args.num_epochs,
        "loss": args.loss_choice,
        "embed": args.embed_dim,
        "heads": args.num_heads,
        "opt": args.optimizer,
        "sched": args.scheduler,
        "aug": "yes" if args.use_aug else "no",
    }

    # Sort the keys to ensure consistent ordering.
    run_name_parts = [f"{key}_{important_args[key]}" for key in sorted(important_args)]
    return "-".join(run_name_parts)

"""
Loss Utils
"""
def get_loss(loss_choice):
    if loss_choice in ["CE"]:
        return nn.CrossEntropyLoss()
    else:
        return nn.MSELoss()


"""
SoftMax Utils
"""
def soft_argmax_coords(value_map: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    value_map: Tensor of shape (b, h, w, 1) containing unnormalized scores.
    tau:       Temperature for softmax; lower = sharper peak.
    
    Returns:
        coords: Tensor of shape (b, 2), where coords[i] = (x_i, y_i) are the
                expected height- and width-indices respectively.
    """
    b, h, w, _ = value_map.shape
    # squeeze the last dim
    scores = value_map.view(b, h, w)
    
    # flatten spatial dims and apply softmax
    flat = scores.view(b, -1)               # (b, h*w)
    probs = F.softmax(flat / tau, dim=-1)   # (b, h*w)
    
    # build coordinate grids
    # grid_h[i,j] = i in [0..h-1], grid_w[i,j] = j in [0..w-1]
    grid_h, grid_w = torch.meshgrid(
        torch.arange(h, device=value_map.device, dtype=torch.float),
        torch.arange(w, device=value_map.device, dtype=torch.float),
        indexing='ij'
    )                                        # both (h, w)
    grid_h = grid_h.reshape(-1)              # (h*w,)
    grid_w = grid_w.reshape(-1)              # (h*w,)
    
    # compute expected coords
    exp_h = (probs * grid_h).sum(dim=-1)     # (b,)
    exp_w = (probs * grid_w).sum(dim=-1)     # (b,)
    
    # stack into (b, 2)
    coords = torch.stack([exp_h, exp_w], dim=-1)
    return coords

"""
Logger utils
"""
class CustomLogger:
    def __init__(self, default_dir="outputs", output_name="run.log"):
        # ensure outputs/ exists
        os.makedirs(default_dir, exist_ok=True)
        self.log_file = open(os.path.join(default_dir, output_name), "a")
        self.terminal = sys.__stdout__  # Always keep the real stdout

    def write(self, message):
        # write to console
        self.terminal.write(message)
        # write to file
        self.log_file.write(message)

    def flush(self):
        # make sure both targets are flushed
        self.terminal.flush()
        self.log_file.flush()
        
    def isatty(self):
        return self.terminal.isatty()

    @property
    def encoding(self):
        return self.terminal.encoding

    def close(self):
        self.log_file.close()
        super().close()