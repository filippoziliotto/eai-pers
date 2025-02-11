# Random Baseline Evaluation

This file presents the performance of two random baselines for predicting a point on a 2D map. Accuracy is evaluated on both training and validation sets at different epochs.
The values 5, 10, and 20 represent distance thresholds used to compute accuracy. A prediction is considered correct if its distance from the ground truth is less than 5, 10, or 20 pixels, resulting in an accuracy of 1; otherwise, the accuracy is 0. In the scene, 1 meter corresponds to a distance of 10 pixels on the map.

### Baseline 1: Random Point at the Center of the Map  
In this baseline, the random prediction is always at the center of the map. The results show that the accuracy remains at 0% for both training and validation across all epochs.

| Epoch | Validation Accuracy |
|-------|--------------------|
| 5     | 0.0000             |
| 10    | 0.0109             |
| 20    | 0.0326             |

### Baseline 1: Random Point where map has non-zero elements
In this baseline, the random prediction is based on the points of the map where the value element is non-zero

| Epoch |Validation Accuracy |
|-------|--------------------|
| 5     | 0.0000             |
| 10    | 0.0000             |
| 20    | 0.0000             |
