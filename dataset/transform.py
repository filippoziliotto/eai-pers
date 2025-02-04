# Custom Transform
import torch
import args  # Import the arguments

class MapTransform:
    """
    Custom transformation class for maps and their corresponding descriptions.
    """
    
    def __init__(self, 
                 use_horizontal_flip=args.USE_HORIZONTAL_FLIP, 
                 use_vertical_flip=args.USE_VERTICAL_FLIP, 
                 use_random_crop=args.USE_RANDOM_CROP,
                 use_desc_aug=args.USE_DESC_AUG,
                 ):
        
        self.use_horizontal_flip = use_horizontal_flip
        self.use_vertical_flip = use_vertical_flip
        self.use_random_crop = use_random_crop

    def __call__(self, feature_map, xy_coords, description):
        """
        Apply transformations to the input data.
        """
        
        # Placeholder for the probability of applying the transformation
        p = 1.
                
        # Apply horizontal flip
        if self.use_horizontal_flip and torch.rand(1) > p:
            feature_map = torch.flip(feature_map, dims=(1,))
            xy_coords[0] = feature_map.shape[1] - xy_coords[0]
            
        # Apply vertical flip
        if self.use_vertical_flip and torch.rand(1) > p:
            feature_map = torch.flip(feature_map, dims=(0,))
            xy_coords[1] = feature_map.shape[0] - xy_coords[1]
        
        # Apply random crop
        if self.use_random_crop and torch.rand(1) > p:
            crop_size = torch.randint(0, 100, (1,)).item()
            feature_map = feature_map[crop_size:-crop_size, crop_size:-crop_size]
            xy_coords[0] -= crop_size
            xy_coords[1] -= crop_size
        
        # Change in description order   
        if self.use_desc_aug and torch.rand(1) > p:
            pass 
        
        return description, feature_map, xy_coords