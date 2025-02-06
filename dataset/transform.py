# Custom Transform
import torch
import args  # Import the arguments
   

class MapTransform:
    """
    Custom transformation class for maps and their corresponding descriptions.
    """
    
    def __init__(self, 
                 **kwargs):
        
        self.use_aug = kwargs.get("use_aug", False)
        self.use_horizontal_flip = kwargs.get("use_horizontal_flip", False)
        self.use_vertical_flip = kwargs.get("use_vertical_flip", False)
        self.use_random_crop = kwargs.get("use_random_crop", False)
        self.use_desc_aug = kwargs.get("use_desc_aug", False)

    def __call__(self, feature_map, xy_coords, description):
        """
        Apply transformations to the input data.
        """
        if not self.use_aug:
            return description, feature_map, xy_coords
                
        # Apply horizontal flip
        if self.use_horizontal_flip and torch.rand(1) > self.prob:
            feature_map = torch.flip(feature_map, dims=(1,))
            xy_coords[0] = feature_map.shape[1] - xy_coords[0]
            
        # Apply vertical flip
        if self.use_vertical_flip and torch.rand(1) > self.prob:
            feature_map = torch.flip(feature_map, dims=(0,))
            xy_coords[1] = feature_map.shape[0] - xy_coords[1]
        
        # Apply random crop
        if self.use_random_crop and torch.rand(1) > self.prob:
            crop_size = torch.randint(0, 100, (1,)).item()
            feature_map = feature_map[crop_size:-crop_size, crop_size:-crop_size]
            xy_coords[0] -= crop_size
            xy_coords[1] -= crop_size
        
        # Change in description order   
        if self.use_desc_aug and torch.rand(1) > self.prob:
            pass 
        
        return description, feature_map, xy_coords