import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union
import wandb

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

def soft_argmax(heatmaps, beta=1.0):
    """
    Compute the soft-argmax of a batch of heatmaps.

    Args:
        heatmaps (torch.Tensor): Tensor of shape (batch_size, height, width).
        beta (float): Temperature parameter to control the sharpness of the softmax.

    Returns:
        coords (torch.Tensor): Tensor of shape (batch_size, 2) containing the (x, y) coordinates.
    """
    batch_size, height, width = heatmaps.shape

    # Create coordinate grid
    # Note: Ensure that the grid is on the same device as the heatmaps.
    grid_y = torch.linspace(0, height - 1, height, device=heatmaps.device)
    grid_x = torch.linspace(0, width - 1, width, device=heatmaps.device)
    # Reshape and expand to match heatmap shape
    grid_y = grid_y.view(1, height, 1).expand(batch_size, height, width)
    grid_x = grid_x.view(1, 1, width).expand(batch_size, height, width)
    
    # Apply the softmax with temperature scaling
    heatmaps = heatmaps * beta
    heatmaps_flat = heatmaps.view(batch_size, -1)
    softmax = F.softmax(heatmaps_flat, dim=1).view(batch_size, height, width)
    
    # Compute expected coordinates
    exp_x = torch.sum(softmax * grid_x, dim=(1, 2))
    exp_y = torch.sum(softmax * grid_y, dim=(1, 2))
    
    # Stack the coordinates together (shape: batch_size x 2)
    coords = torch.stack([exp_x, exp_y], dim=1)
    return coords

"""
Seed Utils
"""
def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
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
Logging Utils
"""
def args_logger(args):
    """
    Logs the arguments to the console.
    
    Args:
        args: The parsed arguments.
    """
    print(' ----------------')
    print("|    Arguments   |")
    print(' ----------------')
    for arg in vars(args):
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

def log_epoch_metrics(epoch, optimizer, train_loss, train_acc, val_loss, val_acc):
    metrics = {
        "Epoch": epoch,
        'Learning Rate': optimizer.param_groups[0]['lr'],
        "Train Loss": train_loss,
        "Val Loss": val_loss,
    }
    metrics.update({f"Train Acc [{k}]": v for k, v in train_acc.items()})
    metrics.update({f"Val Acc [{k}]": v for k, v in val_acc.items()})
    wandb.log(metrics)
    return

"""
Loss Utils
"""
def get_loss(loss_choice):
    if loss_choice in ["CE"]:
        return nn.CrossEntropyLoss()
    else:
        return nn.MSELoss()

def dynamic_min_max_normalize(loss_list, epsilon=1e-8):
    """
    Normalizes a list of loss values using dynamic min-max normalization.

    Args:
        loss_list (List[float]): List of raw loss values.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        List[float]: List of normalized loss values in the range [0, 1].
    """
    if not loss_list:
        return []
    
    min_loss = min(loss_list)
    max_loss = max(loss_list)
    normalized_losses = [
        (loss_val - min_loss) / (max_loss - min_loss + epsilon) for loss_val in loss_list
    ]
    return normalized_losses

"""
Random Baseline
"""
def get_random_target(value_map: torch.Tensor, type='random') -> torch.Tensor:
    """
    For each sample in the batch, returns a random (x, y) coordinate
    where the value in the value_map is non-zero.
    If no non-zero value exists for a sample, a random coordinate is selected.
    
    Args:
        value_map (torch.Tensor): Tensor of shape (batch, width, height)
    
    Returns:
        torch.Tensor: A tensor of shape (batch, 2) with the selected (x, y) coordinates.
    """
    b, w, h = value_map.shape
    pred_target = []

    if type in ['random']:
        for i in range(b):
            # Get indices of non-zero values for sample i.
            nonzero_idx = (value_map[i] != 0).nonzero(as_tuple=False)
            if nonzero_idx.numel() == 0:
                # If no non-zero value is present, select a random coordinate.
                rand_x = torch.randint(0, w, (1,), device=value_map.device).item()
                rand_y = torch.randint(0, h, (1,), device=value_map.device).item()
                pred_target.append(torch.tensor([rand_x, rand_y], device=value_map.device))
            else:
                # Select a random index among the non-zero indices.
                idx = torch.randint(0, nonzero_idx.size(0), (1,)).item()
                pred_target.append(nonzero_idx[idx])
    elif type in ['center']:
        for i in range(b):
            pred_target.append(torch.tensor([w//2, h//2], device=value_map.device))
    else:
        raise ValueError(f"Unsupported type: {type}")
    
    return torch.stack(pred_target, dim=0)