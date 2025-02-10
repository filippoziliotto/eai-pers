# Library imports
from typing import List, Tuple, Dict, Any, Union
from dataset.maps.base_map import BaseMap
from torch.utils.data import random_split, DataLoader

# External imports
import numpy as np
import torch

# Geometry utilities imports
from utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector

   
"""
MAP UTILITIES
"""

def map_to_xyz(episode: Dict, map: BaseMap) -> List[float]:
    """
    Converts 2D map coordinates (x, y) into global habitat coordinates (x, y, z).

    Returns:
        List[float]: The [x, y, z] coordinates in the global frame.
    """
    pos_origin = episode['robot_xyz']
    rotation_world_start = quaternion_from_coeff(episode['robot_rot'])

    # Convert pixel position to local map coordinates and flip y-axis
    local_coords = map._px_to_xy(pos_origin.reshape(1, 2))[0]
    local_coords = np.array([local_coords[0], -local_coords[1]])

    # Create a local vector, rotate it to world coordinates, and translate by the origin
    local_vec = np.array([local_coords[1], 0.0, -local_coords[0]], dtype=np.float32)
    world_vec = quaternion_rotate_vector(rotation_world_start, local_vec)
    return world_vec + pos_origin
    
def xyz_to_map(episode: Dict, map: BaseMap) -> List[float]:
    """
    Converts target coordinates from the global frame (x, y, z) to 2D map frame coordinates (x, y).

    Returns:
        List[float]: The [x, y] coordinates in the map frame.
    """
    # Get rotation from robot heading
    rotation_world_start = quaternion_from_coeff(episode['robot_rot'])

    # Ensure robot_xyz is a NumPy array
    robot_xyz = episode['robot_xyz'] if isinstance(episode['robot_xyz'], np.ndarray) else np.array(episode['robot_xyz'])

    # Rotate target position into the robot's frame of reference
    target_pos = quaternion_rotate_vector(
        rotation_world_start.inverse(), episode['object_pos'] - robot_xyz
    )

    # Convert rotated position to map coordinates (ignoring z and flipping axes)
    target_map = np.array([-target_pos[2], -target_pos[0]], dtype=np.float32)

    # Convert to pixel coordinates on the map
    return map._xy_to_px(target_map.reshape(1, 2))[0]

def load_obstacle_map(path: str) -> np.ndarray:
    """
    Load an obstacle map from a given path.

    Args:
        path (str): The path to the obstacle map file.

    Returns:
        np.ndarray: The obstacle map as a NumPy array.
    """
    return np.load(path).astype(np.int8)
"""
BATCHING UTILITIES
"""
def custom_collate(batch):
    """
    Custom collate function to process a batch of data.
    
    Args:
        batch (list of dict): A list where each element is a dictionary containing
                              'description', 'query', 'target', and 'feature_map'.
    
    Returns:
        dict: A dictionary with keys 'description', 'query', 'target', and 'feature_map'.
              The values are the corresponding stacked tensors or lists from the batch.
    """
    # Extract descriptions and queries from the batch
    descriptions = [item["description"] for item in batch]
    queries = [item["query"] for item in batch]
    
    # Stack the tensors for targets and feature maps
    targets = torch.stack([item["target"] for item in batch])
    feature_maps = torch.stack([item["feature_map"] for item in batch])
    
    # Stack map_path
    map_paths = [item["map_path"] for item in batch]
    
    # Return the collated batch as a dictionary
    return {
        "description": descriptions,
        "query": queries,
        "target": targets,
        "feature_map": feature_maps,
        "map_path": map_paths
    }
    
def split_dataloader(data_loader: DataLoader, split_ratio: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Splits a DataLoader into training and validation DataLoaders based on the given split ratio.
    
    Args:
        data_loader (DataLoader): The original DataLoader to split.
        split_ratio (float): The ratio to split the data into training and validation sets (e.g., 0.8 for 80% train, 20% val).
        batch_size (int): The batch size for the new DataLoaders.
        **kwargs: Additional arguments for the DataLoader.
    
    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders.
    """
    dataset = data_loader.dataset
    total_samples = len(dataset)
    train_size = int(total_samples * split_ratio)
    val_size = total_samples - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

"""
Transforms Utils for Maps
"""
def random_crop_preserving_target(feature_map, xy_coords, p=1.0):
    """
    Randomly crops the feature_map while ensuring that the target point given by xy_coords
    is still within the cropped image.

    Args:
        feature_map (torch.Tensor): A 2D tensor of shape (H, W) representing the feature map.
        xy_coords (list or tuple of int): The target point as [x, y], where x is the column and y is the row.

    Returns:
        cropped_feature_map (torch.Tensor): The cropped feature map.
        new_xy_coords (list of int): The adjusted target coordinates after cropping.
    """

    # Get the height and width of the feature map.
    # This works for a 2D tensor; if your tensor includes channels as the first dimension,
    # adjust accordingly.
    B, H, W, C = feature_map.shape[:2]
    
    # Extract target coordinates (assumed to be [x, y] where x is the horizontal coordinate).
    x, y = xy_coords[0], xy_coords[1]

    # Determine the maximum number of pixels that can be cropped on each side without
    # removing the target point.
    max_crop_left   = x             # Maximum crop from left.
    max_crop_top    = y             # Maximum crop from top.
    max_crop_right  = W - x - 1     # Maximum crop from right.
    max_crop_bottom = H - y - 1     # Maximum crop from bottom.

    # Randomly select crop margins for each side.
    crop_left   = torch.randint(0, max_crop_left + 1, (1,)).item()   if max_crop_left > 0 else 0
    crop_top    = torch.randint(0, max_crop_top + 1, (1,)).item()     if max_crop_top > 0 else 0
    crop_right  = torch.randint(0, max_crop_right + 1, (1,)).item()   if max_crop_right > 0 else 0
    crop_bottom = torch.randint(0, max_crop_bottom + 1, (1,)).item()   if max_crop_bottom > 0 else 0

    # Crop the feature map.
    cropped_feature_map = feature_map[crop_top : H - crop_bottom, crop_left : W - crop_right]

    # Adjust the target coordinates relative to the new cropped image.
    new_xy_coords = [x - crop_left, y - crop_top]

    return cropped_feature_map, new_xy_coords
