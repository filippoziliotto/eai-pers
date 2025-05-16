# Custom Transform
import torch
import random

# Import utility functions
from dataset.utils import random_crop_preserving_target, random_rotate_preserving_target
   
class MapTransform:
    """
    Custom transformation class for maps and their corresponding descriptions.
    """
    
    def __init__(self, augmentations):
        
        self.use_aug = augmentations["use_aug"]
        self.use_horizontal_flip = augmentations["use_horizontal_flip"]
        self.use_vertical_flip = augmentations["use_vertical_flip"]
        self.use_random_crop = augmentations["use_random_crop"]
        self.use_random_rotate = augmentations["use_random_rotate"]
        self.use_desc_aug = augmentations["use_desc_aug"]
        self.prob = augmentations["aug_prob"]
        
        # Angle range for random rotation
        if self.use_random_rotate:
            self.angle_range = (-augmentations["angle_range"], augmentations["angle_range"])

    def __call__(self, feature_map, xy_coords, description):
        """
        Apply transformations to the input data.
        """
        if not self.use_aug:
            return feature_map, xy_coords, description
                
        # Apply horizontal flip (1x more likely)
        if self.use_horizontal_flip and torch.rand(1) < self.prob:
            feature_map = torch.flip(feature_map, dims=(1,))
            xy_coords[0] = feature_map.shape[1] - xy_coords[0]
            
        # Apply vertical flip (1x more likely)
        if self.use_vertical_flip and torch.rand(1) < self.prob:
            feature_map = torch.flip(feature_map, dims=(0,))
            xy_coords[1] = feature_map.shape[0] - xy_coords[1]
            
        # Apply random rotation preserving target if desired.
        if self.use_random_rotate and torch.rand(1) < self.prob:
            feature_map, xy_coords = random_rotate_preserving_target(feature_map, xy_coords, self.angle_range)
    
        # Apply random crop (0.5x more likely)
        if self.use_random_crop and torch.rand(1) < self.prob:
            feature_map, xy_coords = random_crop_preserving_target(feature_map, xy_coords)
        
        # Change in description order (2x more likely)
        if self.use_desc_aug and torch.rand(1) < self.prob*2:
            # Randomly change the order of the elements in the list
            random.shuffle(description)
        
        return feature_map, xy_coords, description