# Library imports
from typing import List, Tuple, Dict
import math
import re

# Torch imports
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


"""
BATCHING UTILITIES
"""
def custom_collate(batch):
    """
    Custom collate function that pads feature maps (which may have been randomly cropped)
    to a common size before stacking them into a batch.
    
    Each feature map is assumed to have shape (H, W, C) (i.e. height, width, channels).
    The padding is applied to the bottom and right so that each feature map reaches the
    maximum height and width found in the batch.
    """
    # Extract descriptions, queries, targets, and map paths.
    summaries = [item["summary"] for item in batch]
    queries = [item["query"] for item in batch]
    targets = torch.stack([torch.tensor(item["target"]) for item in batch])

    # Get the list of feature maps from the batch.
    feature_maps = [item["feature_map"] for item in batch]

    # Check if the feature maps are already padded.
    if not all(fm.shape[:2] == feature_maps[0].shape[:2] for fm in feature_maps):
        # Determine the maximum height and width among all feature maps.
        max_h = max(fm.shape[0] for fm in feature_maps)
        max_w = max(fm.shape[1] for fm in feature_maps)

        padded_feature_maps = []
        for fm in feature_maps:
            h, w, c = fm.shape  # assuming fm has shape (H, W, C)
            # Compute how much to pad on the bottom and right.
            pad_bottom = max_h - h
            pad_right = max_w - w

            # torch.nn.functional.pad expects the padding configuration for the last two dims
            # in the order (pad_left, pad_right, pad_top, pad_bottom). Because our feature map
            # is (H, W, C) (with channels last), we first permute it to (C, H, W) before padding.
            fm_perm = fm.permute(2, 0, 1)  # now shape is (C, H, W)
            # Apply padding: we leave the top and left untouched, so pad_left and pad_top are 0.
            fm_padded = F.pad(fm_perm, (0, pad_right, 0, pad_bottom))
            # Permute back to (H, W, C)
            fm_padded = fm_padded.permute(1, 2, 0)
            padded_feature_maps.append(fm_padded)

        # Now stack the padded feature maps to form a single tensor.
        feature_maps = torch.stack(padded_feature_maps)
        
    else:
        # If the feature maps are already padded, stack them as they are.
        feature_maps = torch.stack(feature_maps)

    return {
        "description": summaries,
        "query": queries,
        "target": targets,
        "feature_map": feature_maps,
    }

"""
Transforms Utils for Maps
"""

def random_crop_preserving_target(feature_map, xy_coords, max_crop_fraction=0.3):
    """
    Randomly crops the feature_map while ensuring that the target point (xy_coords)
    is still within the cropped image. The maximum amount that can be cropped on each side
    is limited to avoid overly drastic crops.

    Args:
        feature_map (torch.Tensor): A tensor of shape (H, W, C) representing the feature map.
        xy_coords (list or tuple of int): The target point as [x, y] (x is the column, y the row).
        max_crop_fraction (float): Maximum fraction of the image dimension to crop on any side.

    Returns:
        cropped_feature_map (torch.Tensor): The cropped feature map.
        new_xy_coords (list of int): The adjusted target coordinates after cropping.
    """
    H, W = feature_map.shape[:2]
    x, y = xy_coords[0], xy_coords[1]

    # Compute the maximum possible crop on each side while keeping the target inside.
    # Here we also subtract 40 pixels on the right and bottom to enforce a minimum
    # remaining size (you can adjust this if needed).
    max_crop_left_possible   = x
    max_crop_top_possible    = y
    max_crop_right_possible  = W - x - 40
    max_crop_bottom_possible = H - y - 40

    # Limit the crop amounts so that we never crop more than max_crop_fraction of the dimension.
    max_crop_left   = int(min(max_crop_left_possible, int(W * max_crop_fraction)))
    max_crop_top    = int(min(max_crop_top_possible, int(H * max_crop_fraction)))
    max_crop_right  = int(min(max_crop_right_possible, int(W * max_crop_fraction)))
    max_crop_bottom = int(min(max_crop_bottom_possible, int(H * max_crop_fraction)))

    # Randomly select crop margins for each side.
    crop_left   = torch.randint(0, max_crop_left + 1, (1,)).item() if max_crop_left > 0 else 0
    crop_top    = torch.randint(0, max_crop_top + 1, (1,)).item() if max_crop_top > 0 else 0
    crop_right  = torch.randint(0, max_crop_right + 1, (1,)).item() if max_crop_right > 0 else 0
    crop_bottom = torch.randint(0, max_crop_bottom + 1, (1,)).item() if max_crop_bottom > 0 else 0

    # Crop the feature map.
    cropped_feature_map = feature_map[crop_top : H - crop_bottom, crop_left : W - crop_right]

    # Adjust the target coordinate relative to the new cropped image.
    new_xy_coords = [x - crop_left, y - crop_top]

    return cropped_feature_map, new_xy_coords

def random_rotate_preserving_target(feature_map, xy_coords, angle_range=(-15, 15)):
    """
    Randomly rotates the feature_map by an angle sampled from angle_range,
    but only applies the rotation if the target coordinate (xy_coords) remains 
    inside the image bounds after rotation. Otherwise, the rotation is skipped.
    
    Args:
        feature_map (torch.Tensor): Tensor of shape (H, W, C).
        xy_coords (list or tuple): The target coordinate as [x, y].
        angle_range (tuple): (min_angle, max_angle) in degrees.
        
    Returns:
        rotated_feature_map (torch.Tensor): Either the rotated feature map or the original feature_map.
        new_xy_coords (list): Updated target coordinate if rotated, otherwise the original xy_coords.
    """
    # Sample a random angle (in degrees) from the provided range.
    angle = torch.empty(1).uniform_(angle_range[0], angle_range[1]).item()
    
    # Get image dimensions and define the center.
    H, W, C = feature_map.shape
    center = (W / 2.0, H / 2.0)
    
    # Compute the new target coordinate using the image-coordinate rotation formula.
    # Note: In image coordinates (origin at top-left, y increases downward),
    # the appropriate transformation when rotating about the center is:
    #   new_dx = dx * cos(theta) + dy * sin(theta)
    #   new_dy = -dx * sin(theta) + dy * cos(theta)
    # where dx = x - center_x and dy = y - center_y.
    x, y = xy_coords
    dx = x - center[0]
    dy = y - center[1]
    
    theta = math.radians(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    new_dx = dx * cos_theta + dy * sin_theta
    new_dy = -dx * sin_theta + dy * cos_theta
    
    new_x = center[0] + new_dx
    new_y = center[1] + new_dy
    
    # Check if the new target coordinate is inside the image bounds.
    if new_x < 0 or new_x >= W or new_y < 0 or new_y >= H:
        # If the target would move outside the image, skip rotation.
        return feature_map, xy_coords
    else:
        # Otherwise, perform the rotation.
        # TF.rotate expects images with channels first, so we permute first.
        fm_perm = feature_map.permute(2, 0, 1)
        rotated_fm = TF.rotate(fm_perm, angle, expand=False, center=center)
        rotated_feature_map = rotated_fm.permute(1, 2, 0)
        return rotated_feature_map, [new_x, new_y]


"""
Naming utils
"""
def count_unique_people(strings: List[str]) -> int:
    """
    Given a list of strings each containing exactly one "<person{number}>",
    returns the count of distinct person IDs.
    """
    pattern = re.compile(r'<person(\d+)>')
    seen = set()

    for s in strings:
        m = pattern.search(s)
        if m:
            seen.add(int(m.group(1)))

    return len(seen)