import torch
import math
import torch.nn as nn
from torch.nn import functional as F

"""
Loss Utils
"""

def compute_loss(ground_truth_coords, value_map, loss_choice='L2'):
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
        return l2_loss(value_map, ground_truth_coords)
    elif loss_choice == 'L1':  # Manhattan loss
        return l1_loss(value_map, ground_truth_coords)
    elif loss_choice == 'CE': # Cross-entropy loss
        return ce_loss(value_map, ground_truth_coords)
    elif loss_choice == 'NCE': # Neighborhood cross-entropy loss
        return neighborhood_ce_loss(value_map, ground_truth_coords)
    else:
        raise ValueError("Invalid loss_choice. Use 'L1' or 'L2'.")


def l2_loss(value_map, target):
    """
    Computes the L2 (Euclidean) loss between predictions and targets.
    """
    #elif loss_choice in ['L1', 'L2']:
    #    # Step 2: Find the predicted coordinates (b, x', y') with max similarity
    #    # Here we choose to return only the max coordinate in a differentiable way.
    #    # You can adjust the temperature to control how "hard" the soft-argmax is.
    #    predicted_coords = []
    #    temperature = 0.1  # Lower values are closer to a hard argmax.
    #    for b in range(value_map.shape[0]):
    #        # Ensure that value_map[b] is a torch.Tensor with requires_grad (it should be if map_features/query_features are)
    #        coords = find_non_zero_neighborhood_indices(
    #            value_map[b], w, neighborhood_size=self.pixels_per_meter//2, return_max=True, temperature=temperature
    #        )
    #        predicted_coords.append(coords)
    #    predicted_coords = torch.stack(predicted_coords)  # Shape: (b, 2)
    #    return predicted_coords
    # TODO:
    #return F.mse_loss(pred, target).sqrt()  # MSE loss and then take the square root
    return


def l1_loss(value_map, target):
    """
    Computes the L1 (Manhattan) loss between predictions and targets.
    """
    # TODO:
    #return F.l1_loss(value_map, target)
    return


def ce_loss(value_map_logits, gt_target):
    """
    Computes the categorical cross-entropy loss between the predicted value_map logits 
    and the ground truth indices.
    
    The second module produces a value_map of shape 
    (batch, w, h, 1) (or (batch, w, h)), which after softmax represents a probability 
    distribution over all spatial positions. The ground_truth_indices should be an index 
    in the range [0, w*h - 1] for each batch element corresponding to the target pixel.
    
    Args:
        value_map_logits (Tensor): Predicted logits of shape (batch, w, h).
        gt_target (Tensor): Ground truth indices of shape (batch, 2).
        
    Returns:
        loss (Tensor): The computed cross entropy loss.
    """
    b, w, h = value_map_logits.size()
    value_map_logits = value_map_logits.view(b, -1)
    assert value_map_logits.dim() == 2, "Value map logits must have shape (batch, w*h)."
    
    # Convert (x, y) coordinates in gt_target to a single index
    x_coords = gt_target[:, 0]  # Shape: (batch,)
    y_coords = gt_target[:, 1]  # Shape: (batch,)
    gt_target = y_coords + x_coords *  h # Shape: (batch,) where we have a single int
    assert gt_target.dim() == 1, "Ground truth indices must be 1D (B,)."
     
    loss = F.cross_entropy(value_map_logits, gt_target)
    return loss

def neighborhood_ce_loss(value_map_logits, gt_target, sigma=1.0):
    """
    Computes the modified cross-entropy loss with a Gaussian weighting to encourage
    predictions near the ground truth pixel.
    
    The loss is penalized more for predictions far from the ground truth, but with a
    Gaussian neighborhood that encourages nearby predictions.
    
    Args:
        value_map_logits (Tensor): Predicted logits of shape (batch, w, h).
        gt_target (Tensor): Ground truth indices of shape (batch, 2).
        w (int): The width of the map.
        h (int): The height of the map.
        sigma (float): The standard deviation for the Gaussian neighborhood weight.
        
    Returns:
        loss (Tensor): The computed weighted cross-entropy loss.
    """
    b, w, h = value_map_logits.size()
    value_map_logits = value_map_logits.view(b, -1)  # Flatten to (batch, w*h)
    
    # Convert (x, y) coordinates in gt_target to a single index
    x_coords = gt_target[:, 0]  # Shape: (batch,)
    y_coords = gt_target[:, 1]  # Shape: (batch,)
    gt_target_index = y_coords + x_coords * h  # Convert to a single index (batch,)
    
    # Create a Gaussian kernel for the neighborhood around the ground truth
    # TODO: change to device
    x = torch.arange(w).float().to("mps")  # Shape: (w, 1)
    y = torch.arange(h).float().to("mps")  # Shape: (1, h)
    X, Y = torch.meshgrid(x, y)  # Shape: (w, h)
    
    # Compute Gaussian weights
    dist = (X - x_coords.unsqueeze(1).unsqueeze(1))**2 + (Y - y_coords.unsqueeze(1).unsqueeze(1))**2
    gaussian_weights = torch.exp(-dist / (2 * sigma**2))
    gaussian_weights = gaussian_weights.view(-1)  # Flatten to (w*h,)
    
    # Normalize the weights so that they sum to 1
    gaussian_weights = gaussian_weights / gaussian_weights.sum()
    
    # Cross-Entropy with Gaussian weights
    log_probs = F.log_softmax(value_map_logits, dim=-1)  # Shape: (batch, w*h)
    weighted_loss = -torch.sum(gaussian_weights * log_probs.gather(1, gt_target_index.unsqueeze(1)))  # Batch-level sum
    
    return weighted_loss.mean()
