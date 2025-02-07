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

def differentiable_soft_argmax(value_map, temperature=1.0):
    """
    Computes a differentiable approximation of argmax using a softmax weighted average.
    
    Parameters:
        value_map (torch.Tensor): A 2D tensor (H x W) of values.
        temperature (float): Temperature parameter to control the sharpness of the softmax.
                             Lower temperatures approximate a hard argmax.
    
    Returns:
        torch.Tensor: A tensor of shape (2,) containing the soft (row, col) coordinates.
    """
    H, W = value_map.shape

    # Compute softmax over the flattened value_map
    value_map_flat = value_map.view(-1)  # shape: (H*W,)
    weights = F.softmax(value_map_flat / temperature, dim=0)  # shape: (H*W,)
    
    # Create coordinate grids for rows and columns
    rows = torch.arange(H, dtype=torch.float32, device=value_map.device)
    cols = torch.arange(W, dtype=torch.float32, device=value_map.device)
    
    # Expand to 2D grids
    # row_grid has shape (H, W) with each row constant
    row_grid = rows.view(H, 1).expand(H, W)
    # col_grid has shape (H, W) with each column constant
    col_grid = cols.view(1, W).expand(H, W)
    
    # Flatten the grids
    row_grid_flat = row_grid.reshape(-1)
    col_grid_flat = col_grid.reshape(-1)
    
    # Compute the expected row and column coordinates
    soft_row = torch.sum(weights * row_grid_flat)
    soft_col = torch.sum(weights * col_grid_flat)
    
    return torch.stack([soft_row, soft_col])

def find_non_zero_neighborhood_indices(value_map, w, neighborhood_size=5, return_max=False, temperature=1.0):
    """
    Finds indices around the maximum value in value_map.
    
    If return_max is True, returns the index of the maximum value.
    In the differentiable version (when return_max is True) we use soft-argmax so that
    the returned coordinates are differentiable.
    
    Parameters:
        value_map (torch.Tensor): 2D tensor of shape (H, W).
        w (int): The width of the value_map (number of columns).
        neighborhood_size (int): Half-size of the neighborhood around the maximum value.
        return_max (bool): If True, return the (row, col) coordinate of the maximum value.
        temperature (float): Temperature for soft-argmax. Lower values approximate a hard argmax.
        
    Returns:
        If return_max is True:
            torch.Tensor: A tensor of shape (2,) with differentiable (row, col) coordinates.
        Else:
            A list of (row, col) tuples corresponding to nonzero values in the neighborhood
            (Note: these indices will be non-differentiable).
    """
    if return_max:
        # Use the differentiable soft-argmax.
        return differentiable_soft_argmax(value_map, temperature=temperature)
    
    # Otherwise, use the discrete approach (non-differentiable)
    # Step 1: Find the index of the maximum value in the value_map
    max_idx = value_map.argmax()
    
    # Step 2: Convert the 1D index to 2D (row, col) coordinates
    max_row, max_col = divmod(int(max_idx), w)
    
    # Calculate the neighborhood boundaries, ensuring they stay within bounds
    row_start = max(0, max_row - neighborhood_size)
    row_end = min(value_map.shape[0], max_row + neighborhood_size + 1)
    col_start = max(0, max_col - neighborhood_size)
    col_end = min(value_map.shape[1], max_col + neighborhood_size + 1)
    
    # Collect indices of non-zero values in the neighborhood
    neighborhood_indices = [
        (r, c)
        for r in range(row_start, row_end)
        for c in range(col_start, col_end)
        if value_map[r, c] != 0
    ]
    
    return neighborhood_indices


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