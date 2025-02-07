
# Library imports
import numpy as np
import torch

def compute_accuracy(gt_target, value_map, thresholds=[5, 10, 20], topk=[1, 3, 5]):
    """
    Computes accuracy, mean squared error, and success rate at different thresholds.

    Args:
        gt_target (list of tuples): List of ground truth coordinates [(x1, y1), (x2, y2), ...].
        value_map (torch.Tensor): Predicted value map of shape (batch, w, h).
        thresholds (list of int): List of distance thresholds to compute success rates (default: [5, 10, 20]).
        topk (list of int): List of top-k max values to compute accuracy (default: [1, 3, 5]).

    Returns:
        dict: Dictionary containing overall accuracy, mean squared error, and success rate at each threshold.
    """
    b, w, h = value_map.size()
    
    # Convert (x, y) coordinates in gt_target to a single index
    x_coords = gt_target[:, 0]  # Shape: (batch,)
    y_coords = gt_target[:, 1]  # Shape: (batch,)
    gt_target = y_coords + x_coords *  h # Shape: (batch,) where we have a single int

    # Send tensors to CPU
    gt_target = gt_target.cpu()
    value_map_flat = value_map.view(b, -1).detach().cpu()
    
    # Top-k Accuracy
    # TODO: Move this into a separate function
    # Initialize dictionary to store Top-k accuracies
    top_k_accuracies = {}
    
    # Loop through each k value (1, 3, 5, etc.)
    for k in topk:
        correct_predictions = 0.0
        
        # Get the top k predicted indices from value_map_flat
        _, top_k_indices = value_map_flat.topk(k, dim=1)  # Shape: (batch, k)
        
        for i in range(value_map_flat.size(0)):  # Iterate over batches
            # Convert predicted indices to (x, y) coordinates
            top_k_x_coords = top_k_indices[i] % w
            top_k_y_coords = top_k_indices[i] // w
            
            # Check if any of the top k predicted indices fall within the neighborhood of the ground truth
            for idx in range(k):
                # Calculate the absolute distance between predicted and ground truth coordinates
                dist_x = torch.abs(top_k_x_coords[idx] - x_coords[i])
                dist_y = torch.abs(top_k_y_coords[idx] - y_coords[i])
                
                # If the distance is within the threshold (10x10 neighborhood), consider it correct
                if dist_x <= 20 and dist_y <= 20:
                    correct_predictions += 1
                    break  # No need to check further for this batch
        
        # Calculate the accuracy for this value of k
        top_k_accuracies[k] = correct_predictions / value_map_flat.size(0)
    

        # MSE Accuracy
        # TODO:
        
    return top_k_accuracies
