# Library imports
import numpy as np
import torch

def compute_accuracy(gt_target, pred_target, thresholds=[5, 10, 20]):
    """
    Computes accuracy, mean squared error, and success rate at different thresholds.

    Args:
        gt_target (list of tuples): List of ground truth coordinates [(x1, y1), (x2, y2), ...].
        pred_target (Dictionary): List of predicted coordinates [(x1, y1), (x2, y2), ...].
        thresholds (list of int): List of distance thresholds to compute success rates (default: [5, 10, 20]).

    Returns:
        dict: Dictionary containing overall accuracy for each threshold.
    """
    pred_target = pred_target['heatmap_coords'].detach().cpu().numpy() 
    gt_target = gt_target.detach().cpu().numpy()
    
    # Compute distances once and then compute mean accuracy for each threshold
    distances = [np.linalg.norm(gt - pred) for gt, pred in zip(gt_target, pred_target)]
    accuracy = {
        threshold: np.mean([int(dist < threshold) for dist in distances])
        for threshold in thresholds
    }

    return accuracy