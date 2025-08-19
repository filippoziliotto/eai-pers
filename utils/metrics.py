# Library imports
import numpy as np
import torch

def compute_accuracy(gt_target, pred_target):
    """
    Computes per‐sample Euclidean distance error and success rates at 1 m, 2 m and 3 m.

    Args:
        gt_target (Tensor): Ground‐truth coordinates, shape (N, 2) in grid indices (each index = 1 m).
        pred_target (dict): Dictionary with key 'coords' containing predicted coordinates, shape (N, 2), floats.

    Returns:
        dict: {
            "distance": list of float,   # Euclidean error (in meters)
            "success_1": float,          # % within 1 m
            "success_2": float,          # % within 2 m
            "success_3": float,          # % within 3 m
        }
    """
    # pull out NumPy arrays
    pred = pred_target['coords'].detach().cpu().numpy()  # (N,2), floats in meters
    gt   = gt_target.detach().cpu().numpy()             # (N,2), can be floats or ints

    # 1) compute Euclidean distances (in meters)
    distances = np.linalg.norm(pred - gt, axis=1) # shape (N,)

    # 2) success rates at 1m, 2m, 3m
    metrics = {"distance": distances.mean().tolist()}
    for r in (1, 2, 3):
        metrics[f"success_{r}"] = np.mean([float(distances[i] <= r) for i in range(len(distances))])

    return metrics
