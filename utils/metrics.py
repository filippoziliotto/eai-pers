# Library imports
import numpy as np
import torch
import matplotlib.pyplot as plt

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

def calculate_metrics_distribution(metrics: list, save_to_disk=False, path_to_img="trainer/visualizations/metrics_distribution.png"):
    """
    Calculates the distribution of metrics.

    Args:
        metrics (dict): List of list of dictionaries containing metrics for each sample.

    Returns:
        dict: Dictionary with the distribution of each metric.
    """
    # histogram of distances
    distances = []
    for epoch_metrics in metrics:
        for batch_metrics in epoch_metrics:
            distances.append(batch_metrics["distance"])

    bins = np.arange(0, 15, 0.5)  # from 0 to 4 meters, step 0.1
    hist, _ = np.histogram(distances, bins=bins, density=True)
    distribution = {
        'distance_bins': bins[:-1],  # exclude the last bin edge
        'distance_distribution': hist.tolist()  # convert to list for easier serialization
    }
    
    if save_to_disk:
        # Save it as a plot
        plt.figure(figsize=(10, 5))
        plt.bar(distribution['distance_bins'], distribution['distance_distribution'], width=0.3, align='edge')
        plt.xlabel('Distance (m)')
        plt.ylabel('Frequency %')
        plt.title('Distribution of Distances')
        plt.xticks(np.arange(0, 15, 1))
        plt.grid(axis='y')
        plt.savefig(path_to_img)
        plt.close()
    
    return distribution