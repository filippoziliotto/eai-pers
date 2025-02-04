import torch
import math


"""
Loss Utils
"""
def l2_loss(pred, target):
    return torch.sqrt(((target - pred) ** 2).sum())

def l1_loss(pred, target):
    return torch.abs(target - pred).sum()

def compute_loss(ground_truth_coords, predicted_coords, loss_choice='L2'):
    """
    Computes the loss between ground truth and predicted coordinates.

    Returns:
        Loss value based on the selected self.loss_choice ('L1' or 'L2').
    """
    
    if loss_choice == 'L2':  # Euclidean loss
        return torch.sqrt(((ground_truth_coords - predicted_coords) ** 2).sum())
    elif loss_choice == 'L1':  # Manhattan loss
        return torch.abs(ground_truth_coords - predicted_coords).sum()

