import numpy as np
import torch
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
    if map_tensor.dim() == 2:
        map_tensor = map_tensor.unsqueeze(-1)
    elif map_tensor.dim() != 3:
        raise ValueError(f"Expected 2D or 3D input, got {map_tensor.dim()}D")
    
    # Reshape using view for better memory efficiency
    h, w, n = map_tensor.shape
    return map_tensor.view(-1, n)

def find_non_zero_neighborhood_indices(value_map, w, neighborhood_size=10):
    """
    Finds the indices of non-zero values in the neighborhood around the maximum value in the value_map.

    Parameters:
    value_map (numpy.ndarray): 2D array of values where we want to find the maximum similarity.
    w (int): Width of the value_map.
    neighborhood_size (int): Size of the neighborhood to consider around the maximum value. Default is 10.

    Returns:
    list of tuple: List of indices (row, col) of non-zero values in the neighborhood around the maximum value.
    """
    # Step 1: Find the index of the maximum value in the value_map
    max_idx = value_map.argmax()

    # Step 2: Calculate the row and column indices of the maximum value
    max_row, max_col = divmod(max_idx, w)

    # Step 3: Define the neighborhood boundaries
    row_start = max(0, max_row - neighborhood_size)
    row_end = min(value_map.shape[0], max_row + neighborhood_size)
    col_start = max(0, max_col - neighborhood_size)
    col_end = min(value_map.shape[1], max_col + neighborhood_size)

    # Step 4: Extract the neighborhood around the maximum value
    neighborhood = value_map[row_start:row_end, col_start:col_end]

    # Step 5: Find the indices of non-zero values in the neighborhood
    non_zero_indices = [(i + row_start, j + col_start) for i, row in enumerate(neighborhood) for j, val in enumerate(row) if val != 0]

    # Output: List of tuples (row, col)
    return non_zero_indices

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