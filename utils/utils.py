import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union

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


def get_loss(loss_choice):
    if loss_choice in ["CE"]:
        return nn.CrossEntropyLoss()
    else:
        return nn.MSELoss()

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