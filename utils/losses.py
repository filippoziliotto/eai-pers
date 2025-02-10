
# Library imports
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

# Local imports
from utils.utils import soft_argmax

"""
Loss Utils
"""

def compute_loss(ground_truth_coords, value_map, loss_choice='L2', device="cuda"):
    """
    Computes the loss between ground truth and predicted coordinates.
    
    Args:
        ground_truth_coords (Tensor): Ground truth coordinates.
        predicted_coords (Tensor): Predicted coordinates.
        loss_choice (str): Either 'L1' for Manhattan loss or 'L2' for Euclidean loss.
        
    Returns:
        loss (Tensor): Computed loss.
    """
    if loss_choice == 'MSE':  # Euclidean loss
        return mse_loss(value_map, ground_truth_coords)
    else:
        raise ValueError("Invalid loss_choice. Use only implemented losses.")

def mse_loss(value_map, target):
    """
    Computes the L2 (Euclidean) loss between predictions and targets.
    """
    predicted_coords = soft_argmax(value_map, beta=10.0)
    return F.mse_loss(predicted_coords, target.to(torch.float32)), predicted_coords