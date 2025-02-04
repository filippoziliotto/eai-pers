# Library imports
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
    
# Local imports
from datasets.load_maps import load_episodes
from datasets.utils import xyz_to_map

# Local Map imports
from datasets.maps.base_map import BaseMap
from datasets.transform import MapTransform


class RetMapsDataset(Dataset):
    """
    Dataset class for retrieving maps and their corresponding descriptions.
    Load everything in memory to train/eval
    """
    base_map = None
    
    def __init__(self, data_dir="data", data_split="val", transform=None):
        self.episodes = load_episodes(data_dir, data_split)
        self.episodes_dir = os.path.join(data_dir, data_split)
        
        self.transform = transform

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        
        # Get episode data
        episode = self.episodes[idx]
        
        # Get episode Description
        description = episode["description"]
        
        # Uncompress the feature map (.npz)
        feature_map = np.load(episode["feature_map_path"])
        feature_map = torch.tensor(feature_map["arr_0"])

        # This is the y label
        xy_target = xyz_to_map(episode["object_pos"], episode["robot_xyz"], episode["robot_xy"], episode["robot_heading"])
        
        # Transform the data, used only for convert to tensor or augmentations
        if self.transform:
            description, feature_map, xy_target = self.transform(feature_map, xy_target, episode["description"])
            
        return description, feature_map, xy_target

def get_dataloader(data_dir, data_split="val", batch_size=32, shuffle=True, num_workers=4, **kwargs):
    """
    Creates a dataloader with optional transformation arguments.

    Parameters:
        data_dir (str): Path to the dataset.
        data_split (str): Dataset split ("train", "val", "test").
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for data loading.
        **kwargs: Additional keyword arguments for MapTransform.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    transform = MapTransform(**kwargs)  # Pass all extra arguments to MapTransform
    
    dataset = RetMapsDataset(data_dir, data_split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader