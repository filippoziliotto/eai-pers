
# Library imports
from torch.nn import functional as F
import torch.nn as nn

"""
Loss Utils
"""

def compute_loss(gt_coords, output, loss_choice='L2'):
    """
    Computes the loss between ground truth and predicted coordinates.
    
    Args:
        ground_truth_coords (Tensor): Ground truth coordinates.
        predicted_coords (Tensor): Predicted coordinates.
        loss_choice (str): Either 'L1' for Manhattan loss or 'L2' for Euclidean loss.
        
    Returns:
        loss (Tensor): Computed loss.
    """
    if loss_choice == 'L1':
        loss = L1_loss(output["coords"], gt_coords)
    elif loss_choice == 'L2':
        loss = L2_loss(output["coords"], gt_coords)
    elif loss_choice == 'Huber':
        loss = Huber_loss(output["coords"], gt_coords)
    else:
        raise ValueError(f"Unknown loss choice: {loss_choice}. Use 'L1' or 'L2'.")
    
    return loss
    
    
def L1_loss(pred_coords, gt_coords):
    """
    Computes Mean Absolute Error (L1) loss between the predicted coordinates 
    from the global regression branch and the ground truth coordinates.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (b, 2).
        gt_coords (torch.Tensor): Ground truth coordinates of shape (b, 2).
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    return F.l1_loss(pred_coords, gt_coords)

def L2_loss(pred_coords, gt_coords):
    """
    Computes Mean Squared Error (L2) loss between the predicted coordinates 
    from the global regression branch and the ground truth coordinates.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (b, 2).
        gt_coords (torch.Tensor): Ground truth coordinates of shape (b, 2).
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    return F.mse_loss(pred_coords, gt_coords)

def Huber_loss(pred_coords, gt_coords, delta=1.0):
    """
    Computes Huber loss between the predicted coordinates and ground truth coordinates.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (b, 2).
        gt_coords (torch.Tensor): Ground truth coordinates of shape (b, 2).
        delta (float): The point where the loss function changes
            from quadratic to linear. Default is 1.0.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    return nn.SmoothL1Loss(beta=delta)(pred_coords, gt_coords)
