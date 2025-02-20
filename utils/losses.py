
# Library imports
from torch.nn import functional as F

"""
Loss Utils
"""

def compute_loss(gt_coords, output, loss_choice='L2', scaling = 0.5, device="cuda"):
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
        return regression_loss(gt_coords, output['regression_coords'])
    elif loss_choice == 'HYBRID':  # Manhattan loss
        reg_loss = regression_loss(gt_coords, output['regression_coords'])
        hm_loss = heatmap_loss(gt_coords, output['heatmap_coords'])
        return scaling * reg_loss + (1 - scaling) * hm_loss
    else:
        raise ValueError("Invalid loss_choice. Use only implemented losses.")


def regression_loss(pred_coords, gt_coords):
    """
    Computes Mean Squared Error (MSE) loss between the predicted coordinates 
    from the global regression branch and the ground truth coordinates.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (b, 2).
        gt_coords (torch.Tensor): Ground truth coordinates of shape (b, 2).
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    return F.mse_loss(pred_coords, gt_coords)

def heatmap_loss(pred_coords, gt_coords):
    """
    Computes Mean Squared Error (MSE) loss between the predicted coordinates 
    from the soft-argmax heatmap branch and the ground truth coordinates.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (b, 2).
        gt_coords (torch.Tensor): Ground truth coordinates of shape (b, 2).
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    return F.mse_loss(pred_coords, gt_coords)