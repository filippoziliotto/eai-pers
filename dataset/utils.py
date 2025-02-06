# Library imports
from typing import List, Tuple, Dict, Any, Union
from dataset.maps.base_map import BaseMap
from torch.utils.data import random_split, DataLoader

# External imports
import numpy as np

# Geometry utilities imports
from utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector


def map_to_xyz(episode: Dict, map: BaseMap) -> List[float]:
    """
    Converts 2D map coordinates (x, y) into global habitat coordinates (x, y, z).

    Returns:
        List[float]: The [x, y, z] coordinates in the global frame.
    """
    pos_origin = episode['robot_xyz']
    rotation_world_start = quaternion_from_coeff(episode['robot_heading'])

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
    rotation_world_start = quaternion_from_coeff(episode['robot_heading'])

    # Ensure robot_xyz is a NumPy array
    robot_xyz = episode['robot_xyz'] if isinstance(episode['robot_xyz'], np.ndarray) else np.array(episode['robot_xyz'])

    # Rotate target position into the robot's frame of reference
    target_pos = quaternion_rotate_vector(
        rotation_world_start.inverse(), episode['xyz_target'] - robot_xyz
    )

    # Convert rotated position to map coordinates (ignoring z and flipping axes)
    target_map = np.array([-target_pos[2], -target_pos[0]], dtype=np.float32)

    # Convert to pixel coordinates on the map
    return map._xy_to_px(target_map.reshape(1, 2))[0]

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