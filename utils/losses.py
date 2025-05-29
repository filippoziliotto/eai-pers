
# Library imports
from torch.nn import functional as F
import torch.nn as nn
import torch

# Local imports
from utils.heatmap import generate_gt_heatmap

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
    
    if "coords" in output.keys():
        # We use the normale L2 loss
        return L2_loss(output["coords"].to(torch.float32), gt_coords)
    
    if loss_choice == 'L1':
        return L1_loss(output["coords"], gt_coords)
    elif loss_choice == 'L2':
        return L2_loss(output["coords"], gt_coords)
    elif loss_choice == 'Huber':
        return Huber_loss(output["coords"], gt_coords)
    elif loss_choice == 'Huber+Heatmap':
        # Compute the GT heatmap (b, H, W, 1)
        gt_heatmap = generate_gt_heatmap(gt_coords, output["value_map"])
        pred_heatmap = output["value_map"]
        
        return Huber_loss(output["coords"], gt_coords) + Heatmap_loss(pred_heatmap, gt_heatmap)
    elif loss_choice == 'Heatmap':
        # Compute the GT heatmap (b, H, W, 1)
        gt_heatmap = generate_gt_heatmap(gt_coords, output["value_map"])
        pred_heatmap = output["value_map"]
        
    elif loss_choice == 'SCE':
        # Scaled Cross-Entropy loss
        gt_heatmap = generate_gt_heatmap(gt_coords, output["value_map"])
        pred_heatmap = output["value_map"]
        return ScaledCE_loss(pred_heatmap, gt_heatmap, output["dist_matrix"])
    else:
        raise ValueError(f"Unknown loss choice: {loss_choice}. Use 'L1' or 'L2', 'Huber'.")
    
    
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

def Heatmap_loss(pred_heatmap: torch.Tensor,
                 gt_heatmap: torch.Tensor,
                 reduction: str = 'mean') -> torch.Tensor:
    """
    Computes MSE loss between predicted heatmap and ground-truth heatmap.

    Args:
        pred_heatmap (Tensor): Predicted heatmap, shape (b, H, W, 1) or (b, 1, H, W).
        gt_heatmap   (Tensor): Ground-truth heatmap, same shape as pred.
        reduction    (str): ‘mean’ | ‘sum’ | ‘none’. How to reduce per-pixel losses.

    Returns:
        Tensor: Scalar loss if reduction!='none', else loss map of shape (b, H, W, 1).
    """
    # ensure shape (b, 1, H, W)
    if pred_heatmap.dim() == 4 and pred_heatmap.shape[-1] == 1:
        pred = pred_heatmap.permute(0, 3, 1, 2)
        gt   = gt_heatmap.permute(0, 3, 1, 2)
    else:
        pred, gt = pred_heatmap, gt_heatmap

    # MSE between the two heatmaps
    loss = F.mse_loss(pred, gt, reduction=reduction)
    return loss


def ScaledCE_loss(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    dist_matrix: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes a scaled cross-entropy loss between predicted and ground truth heatmaps.

    Args:
        pred_heatmap (torch.Tensor): Predicted heatmap tensor of shape (N, H, W, 1) or (N, H, W).
        gt_heatmap (torch.Tensor): Ground truth heatmap tensor of shape (N, H, W, 1) or (N, H, W).
        dist_matrix (torch.Tensor): Distance matrix tensor of shape (N, H, W, 1) or (N, H, W).
        reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum'.

    Returns:
        torch.Tensor: The computed scaled cross-entropy loss (scalar).
    """

    # Squeeze to (n, H, W) and Flatten the matrices
    pred_heatmap_flat = pred_heatmap.squeeze(-1).view(pred_heatmap.size(0), -1)
    gt_heatmap_flat = gt_heatmap.squeeze(-1).view(gt_heatmap.size(0), -1)
    dist_matrix_flat = dist_matrix.squeeze(-1).view(dist_matrix.size(0), -1)
    
    # Cross-Entropy loss manually
    # ce_loss = gt_heatmap_flat * F.log_softmax(pred_heatmap_flat, dim=1)
    
    # Binary cross entropy with reduction none
    ce_loss = F.binary_cross_entropy_with_logits(pred_heatmap_flat, gt_heatmap_flat, reduction='none')
    
    
    # Max between dist_matrix and gt_heatmap
    max_dist = torch.max(dist_matrix_flat, gt_heatmap_flat)
    
    # Scale the CE loss by the max distance
    scaled_ce_loss = torch.sum(ce_loss * max_dist, dim=-1)
    
    if reduction == 'mean':
        return scaled_ce_loss.mean()
    elif reduction == 'sum':
        return scaled_ce_loss.sum()
    else:
        raise ValueError(f"Unknown reduction method: {reduction}. Use 'mean' or 'sum'.")
    