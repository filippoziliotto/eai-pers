import torch
import math
import torch.nn as nn
from torch.nn import functional as F

"""
Loss Utils
"""

def compute_loss(ground_truth_coords, predicted_coords, loss, loss_choice='L2'):
    """
    Computes the loss between ground truth and predicted coordinates.
    
    Args:
        ground_truth_coords (Tensor): Ground truth coordinates.
        predicted_coords (Tensor): Predicted coordinates.
        loss_choice (str): Either 'L1' for Manhattan loss or 'L2' for Euclidean loss.
        
    Returns:
        loss (Tensor): Computed loss.
    """
    if loss_choice == 'L2':  # Euclidean loss
        return l2_loss(predicted_coords, ground_truth_coords)
    elif loss_choice == 'L1':  # Manhattan loss
        return l1_loss(predicted_coords, ground_truth_coords)
    elif loss_choice == 'CE': # Cross-entropy loss
        assert ground_truth_coords.dim() == 2, "Ground truth indices must be 2D (B,)."
        return compute_cross_entropy_loss(predicted_coords, ground_truth_coords, loss)
    else:
        raise ValueError("Invalid loss_choice. Use 'L1' or 'L2'.")


def l2_loss(pred, target):
    """
    Computes the L2 (Euclidean) loss between predictions and targets.
    """
    return F.mse_loss(pred, target).sqrt()  # MSE loss and then take the square root


def l1_loss(pred, target):
    """
    Computes the L1 (Manhattan) loss between predictions and targets.
    """
    return F.l1_loss(pred, target)


def compute_cross_entropy_loss(value_map_logits, ground_truth_indices, loss_fn):
    """
    Computes the categorical cross-entropy loss between the predicted value_map logits 
    and the ground truth indices.
    
    The second module produces a value_map of shape 
    (batch, w, h, 1) (or (batch, w, h)), which after softmax represents a probability 
    distribution over all spatial positions. The ground_truth_indices should be an index 
    in the range [0, w*h - 1] for each batch element corresponding to the target pixel.
    
    Args:
        value_map_logits (Tensor): Predicted logits of shape (batch, w, h, 1) or (batch, w, h).
        ground_truth_indices (Tensor): Ground truth indices of shape (batch,).
        
    Returns:
        loss (Tensor): The computed cross entropy loss.
    """
    batch = value_map_logits.size(0)
    
    # If the logits have a trailing singleton dimension, remove it.
    if value_map_logits.dim() == 4 and value_map_logits.size(-1) == 1:
        value_map_logits = value_map_logits.squeeze(-1)  # Now shape: (batch, w, h)
    
    # Flatten the spatial dimensions to shape (batch, w*h)
    value_map_logits = value_map_logits.view(batch, -1)
    
    loss = loss_fn(value_map_logits, ground_truth_indices)
    return loss
