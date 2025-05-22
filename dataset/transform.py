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
        
        # Use of augmentations
        self.use_aug = augmentations["use_aug"]
        self.default_prob = augmentations["default_prob"]
        
        # H/V Flip augmentation
        self.use_horizontal_flip = augmentations["flip"]["use_horizontal_flip"]
        self.use_vertical_flip = augmentations["flip"]["use_vertical_flip"]
        if self.use_horizontal_flip or self.use_vertical_flip:
            self.flip_prob = augmentations["flip"]["prob"]
        
        # Description augmentation
        self.use_desc_aug = augmentations["desc"]["use_desc"]
        if self.use_desc_aug:
            self.desc_prob = augmentations["desc"]["prob"]
        
        # Random crop augmentation
        self.use_random_crop = augmentations["crop"]["use_crop"]
        if self.use_random_crop:
            self.max_crop_fraction = augmentations["crop"]["max_crop_fraction"]
            self.crop_prob = augmentations["crop"]["prob"]
        
        # Random rotation augmentation
        self.use_random_rotate = augmentations["rotation"]["use_rotation"]
        if self.use_random_rotate:
            self.angle_range = (
                -augmentations["rotation"]["angle_range"], 
                augmentations["rotation"]["angle_range"]
            )
            self.rot_prob = augmentations["rotation"]["prob"]

    def __call__(self, feature_map, xy_coords, description):
        """
        Apply transformations to the input data.
        """
        if not self.use_aug:
            return feature_map, xy_coords, description
                
        # Apply horizontal flip
        if self.use_horizontal_flip and torch.rand(1) < self.flip_prob:
            feature_map = torch.flip(feature_map, dims=(1,))
            xy_coords[0] = feature_map.shape[1] - xy_coords[0]
            
        # Apply vertical flip
        if self.use_vertical_flip and torch.rand(1) < self.flip_prob:
            feature_map = torch.flip(feature_map, dims=(0,))
            xy_coords[1] = feature_map.shape[0] - xy_coords[1]
            
        # Apply random rotation
        if self.use_random_rotate and torch.rand(1) < self.rot_prob:
            feature_map, xy_coords = random_rotate_preserving_target(feature_map, xy_coords, self.angle_range)
    
        # Apply random crop
        if self.use_random_crop and torch.rand(1) < self.crop_prob:
            feature_map, xy_coords = random_crop_preserving_target(feature_map, xy_coords, self.max_crop_fraction)
        
        # Change description order elements
        if self.use_desc_aug and torch.rand(1) < self.desc_prob:
            random.shuffle(description)
        
        return feature_map, xy_coords, description