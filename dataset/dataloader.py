# Library imports
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# Add base path to PYTHONPATH
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)
    
# Local imports
from dataset.load_maps import load_episodes, load_extracted_episodes
from dataset.utils import xyz_to_map

# Local Map imports
from dataset.maps.base_map import BaseMap
from dataset.transform import MapTransform

USE_EXTRACTOR = True

class RetMapsDataset(Dataset):
    """
    Dataset class for retrieving maps and their corresponding descriptions.
    Load everything in memory to train/eval
    """
    map = BaseMap(size=500, pixels_per_meter=10)
    
    def __init__(self, data_dir="data", data_split="val", transform=None):
        
        if USE_EXTRACTOR:
            self.episodes = load_extracted_episodes(data_dir, data_split)
        else:
            self.episodes = load_episodes(data_dir, data_split)
        self.episodes_dir = os.path.join(data_dir, data_split)
        
        self.transform = transform

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        
        # Get episode data
        episode = self.episodes[idx]
        
        # Get episode Description
        description = episode["summary_extraction"]
        query = episode["query"]
        
        # Uncompress the feature map (.npz)
        feature_map = np.load(episode["feature_map_path"])
        feature_map = torch.tensor(feature_map["arr_0"])

        # This is the y label
        target = xyz_to_map(episode, self.map)

        # Transform the data, used only for convert to tensor or augmentations
        if self.transform:
            feature_map, target, description = self.transform(feature_map, target, description)
            
        # Return as a dictionary
        return {
            "description": description,
            "target": target,
            "query": query,
            "feature_map": feature_map
        }

def get_dataloader(data_dir, data_split="val", batch_size=32, shuffle=True, num_workers=4, collate_fn=None, kwargs={}):
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
    print("Initializing DataLoader...")
    
    transform = MapTransform(**kwargs)  # Pass all extra arguments to MapTransform
    
    dataset = RetMapsDataset(data_dir, data_split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    print("DataLoader initialized.")
    
    return dataloader


if __name__ == "__main__":
    # Test the dataloader
    dataloader = get_dataloader("data", data_split="val", batch_size=2, shuffle=True, num_workers=4)
    
    for i, episode in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"Query: {episode['query']}")
        print(f"Description: {episode['description']}")
        print(f"Feature Map: {episode['feature_map'].shape}")
        print(f"XY Target: {episode['target']}")
        print()
        
        if i == 0:
            break