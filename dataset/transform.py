# Custom Transform
import torch
import args  # Import the arguments

DEBUG = True
if DEBUG:
    args.USE_AUG = False
    args.AUG_PROB = 0.5
    args.USE_HORIZONTAL_FLIP = True
    args.USE_VERTICAL_FLIP = True
    args.USE_RANDOM_CROP = True
    args.USE_DESC_AUG = True
    

class MapTransform:
    """
    Custom transformation class for maps and their corresponding descriptions.
    """
    
    def __init__(self, 
                 use_aug=args.USE_AUG,
                 prob=args.AUG_PROB,
                 use_horizontal_flip=args.USE_HORIZONTAL_FLIP, 
                 use_vertical_flip=args.USE_VERTICAL_FLIP, 
                 use_random_crop=args.USE_RANDOM_CROP,
                 use_desc_aug=args.USE_DESC_AUG,
                 ):
        
        self.use_aug = use_aug
        self.prob = prob
        self.use_horizontal_flip = use_horizontal_flip
        self.use_vertical_flip = use_vertical_flip
        self.use_random_crop = use_random_crop
        self.use_desc_aug = use_desc_aug

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