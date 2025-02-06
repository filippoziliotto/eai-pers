
# Library imports
import numpy as np

def accuracy(gt_points, pred_points, threshold=10):
    """
    Computes accuracy for a batch of points.
    
    Args:
        gt_points (np.ndarray): Ground truth coordinates of shape (N, 2).
        pred_points (np.ndarray): Predicted coordinates of shape (N, 2).
        threshold (int): Distance threshold for considering the prediction correct (default is 10).
    
    Returns:
        np.ndarray: Array of 1s and 0s indicating whether each prediction is correct.
    """
    assert gt_points.shape == pred_points.shape, "Ground truth and predictions must have the same shape."
    distances = np.linalg.norm(gt_points - pred_points)
    return (distances <= threshold).astype(int)

def mean_squared_error(gt_point, pred_point):
    """
    Computes the Mean Squared Error (MSE) between the ground truth point and the predicted point.

    Args:
        gt_point (tuple): Ground truth coordinates (x, y).
        pred_point (tuple): Predicted coordinates (x, y).

    Returns:
        float: Mean squared error.
    """
    error = np.array(gt_point) - np.array(pred_point)
    mse = np.mean(error ** 2)
    return mse

def success_rate(gt_points, pred_points, thresholds=[5, 10, 20]):
    """
    Computes the success rate at multiple distance thresholds.

    Args:
        gt_points (list of tuples): List of ground truth coordinates [(x1, y1), (x2, y2), ...].
        pred_points (list of tuples): List of predicted coordinates [(x1, y1), (x2, y2), ...].
        thresholds (list of int): List of distance thresholds to compute success rates (default: [5, 10, 20]).

    Returns:
        dict: Success rate for each threshold, e.g., {'5': 0.8, '10': 0.9, '20': 1.0}.
    """
    assert len(gt_points) == len(pred_points), "Ground truth and predictions must have the same length."
    return {
        str(threshold): sum(accuracy(gt, pred, threshold) for gt, pred in zip(gt_points, pred_points)) / len(gt_points)
        for threshold in thresholds
    }

def compute_accuracy(gt_points, pred_points, thresholds=[5, 10, 20]):
    """
    Computes accuracy, mean squared error, and success rate at different thresholds.

    Args:
        gt_points (list of tuples): List of ground truth coordinates [(x1, y1), (x2, y2), ...].
        pred_points (list of tuples): List of predicted coordinates [(x1, y1), (x2, y2), ...].
        thresholds (list of int): List of distance thresholds to compute success rates (default: [5, 10, 20]).

    Returns:
        dict: Dictionary containing overall accuracy, mean squared error, and success rate at each threshold.
    """
    assert len(gt_points) == len(pred_points), "Ground truth and predictions must have the same length."

    # Send tensors to CPU and convert to numpy arrays
    gt_points = [gt.detach().cpu().numpy() for gt in gt_points]
    pred_points = [pred.detach().cpu().numpy() for pred in pred_points]

    # TODO: Fix this line   
    # Compute mean squared error for all points
    # mse = torch.mean([mean_squared_error(gt, pred) for gt, pred in zip(gt_points, pred_points)])
    
    # Compute success rate at different thresholds
    success_rates = success_rate(gt_points, pred_points, thresholds)
    
    # TODO: return all metrics, for now we return only the success rates for each threshold
    return success_rates
