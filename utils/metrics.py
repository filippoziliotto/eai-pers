import numpy as np

def accuracy(gt_point, pred_point, threshold=10):
    """
    Computes accuracy based on whether the predicted point is within a specified threshold of the ground truth point.

    Args:
        gt_point (tuple): Ground truth coordinates (x, y).
        pred_point (tuple): Predicted coordinates (x, y).
        threshold (int): Distance threshold for considering the prediction correct (default is 10).

    Returns:
        int: 1 if the prediction is within the threshold distance, 0 otherwise.
    """
    distance = np.linalg.norm(np.array(gt_point) - np.array(pred_point))
    return 1 if distance <= threshold else 0

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
    success_rates = {}

    for threshold in thresholds:
        successes = sum(accuracy(gt, pred, threshold) for gt, pred in zip(gt_points, pred_points))
        success_rates[str(threshold)] = successes / len(gt_points)
    
    return success_rates

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

    # Compute overall accuracy with the default threshold of 10
    total_accuracy = sum(accuracy(gt, pred) for gt, pred in zip(gt_points, pred_points)) / len(gt_points)
    
    # Compute mean squared error for all points
    mse = np.mean([mean_squared_error(gt, pred) for gt, pred in zip(gt_points, pred_points)])
    
    # Compute success rate at different thresholds
    success_rates = success_rate(gt_points, pred_points, thresholds)
    
    # TODO: return all metrics, for now we return only the sr
    return success_rates
