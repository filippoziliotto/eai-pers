# Library imports
import numpy as np
import torch

def compute_accuracy_old(gt_target, pred_target):
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
    distances = np.linalg.norm(pred - gt, axis=1)  # shape (N,)

    # 2) success rates at 1m, 2m, 3m
    metrics = {"distance": distances.mean().tolist()}
    for r in (1, 2, 3):
        metrics[f"success_{r}"] = np.mean([float(distances[i] <= r) for i in range(len(distances))])

    return metrics

def compute_accuracy(gt_coords: torch.Tensor,
                        output: torch.Tensor) -> dict:
    """
    Args:
        gt_coords: Tensor of shape (b,2) with ground-truth (y, x) pixel indices.
        value_map: Tensor of shape (b, H, W, 1) of raw scores per pixel.

    Returns:
        dict with keys:
          - "distance" : float, mean Euclidean error in pixels
          - "success_1": float, fraction with error ≤ 1 px
          - "success_2": float, fraction with error ≤ 2 px
          - "success_3": float, fraction with error ≤ 3 px
    """
    
    # This is uses for the baselines
    if "coords" in output.keys():
        return compute_accuracy_old(gt_coords, output)
    
    
    value_map = output['value_map']  # (b, H, W, 1)
    b, H, W, _ = value_map.shape

    # 1) Flatten spatial dims and find arg-max per sample
    flat = value_map.view(b, -1)           # (b, H*W)
    idx = torch.argmax(flat, dim=1)        # (b,) linear index

    # 2) Convert linear index to (y, x)
    y_pred = (idx // W).float()            # (b,)
    x_pred = (idx %  W).float()            # (b,)
    preds  = torch.stack([y_pred, x_pred], dim=1)  # (b,2)

    # 3) Euclidean distances
    diffs = preds - gt_coords              # (b,2)
    dists = torch.norm(diffs, dim=1)       # (b,)

    # 4) Aggregate metrics
    distances = dists.detach().cpu().numpy()
    mean_dist = float(distances.mean())

    metrics = {"distance": mean_dist}
    for r in (1, 2, 3):
        metrics[f"success_{r}"] = float((distances <= r).mean())

    return metrics