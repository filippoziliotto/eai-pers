# Library imports
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import os

# Add base path to PYTHONPATH
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)
    
# Local imports
from dataset.load_maps import load_episodes, load_extracted_episodes
from dataset.utils import xyz_to_map
from dataset.maps.base_map import BaseMap
from dataset.transform import MapTransform

# Config
import config

class RetMapsDataset(Dataset):
    """
    Dataset class for retrieving maps and their corresponding descriptions.
    Load everything in memory to train/eval
    """
    map = BaseMap(size=500, pixels_per_meter=10)
    
    def __init__(self, data_dir="data", data_split="val", transform=None):
        
        if config.USE_EXTRACTOR:
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
        
        # Path to obstacle_map
        # TODO: save augmentations type and add to obstacle_map
        map_path = episode["feature_map_path"].split("feature_map.npz")[0]

        # This is the gt position of the object in map frame
        target = xyz_to_map(episode, self.map)

        # Transform the data, used only for convert to tensor or augmentations
        if self.transform:
            feature_map, target, description = self.transform(feature_map, target, description)
            
        # Return as a dictionary
        return {
            "description": description,
            "target": target,
            "query": query,
            "feature_map": feature_map,
            "map_path": map_path
        }

def get_dataloader(data_dir, data_split="train+val", batch_size=32, num_workers=4, 
                   collate_fn=None, shuffle=None, kwargs={}):
    """
    Returns DataLoaders based on the specified data_split.
    
    Modes:
      - "val": Use only the validation set (from data_dir/"val"), and split it into:
              - A training subset (80%) with augmentations applied if kwargs["use_aug"] is True.
              - A validation subset (20%) with augmentations always disabled.
              
      - "train+val": Load separate datasets:
              - Training dataset from data_dir/"train" (using kwargs["use_aug"] as provided).
              - Validation dataset from data_dir/"val" with augmentations disabled.
    
    Parameters:
      data_dir (str): Path to the dataset.
      data_split (str): Either "val" or "train+val".
      batch_size (int): Batch size.
      num_workers (int): Number of DataLoader workers.
      collate_fn (callable): Custom collate function.
      shuffle (bool): (Not used here; DataLoader shuffle is set internally: train=True, val=False)
      kwargs (dict): Additional keyword arguments for MapTransform.
      
    Returns:
      Tuple (train_loader, val_loader)
    """
    print("Initializing DataLoader...")
    
    if data_split == "val":
        print("Using validation dataset only. Splitting into training and validation subsets.")
               
        # --- Determine the split indices ---
        # Load a temporary dataset without any transformation to get the full list of episodes.
        temp_dataset = RetMapsDataset(data_dir, "val", transform=None)
        total_samples = len(temp_dataset)
        train_size = int(0.8 * total_samples)
        indices = list(range(total_samples))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # --- Prepare transform parameters ---
        # For the training subset, honor the provided use_aug flag (defaulting to True).
        kwargs_train = kwargs.copy()
        kwargs_train["use_aug"] = kwargs.get("use_aug", True)
        
        # For the validation subset, always disable augmentation.
        kwargs_val = kwargs.copy()
        kwargs_val["use_aug"] = False
        
        # --- Create two separate dataset instances with different transforms ---
        train_dataset_full = RetMapsDataset(data_dir, "val", transform=MapTransform(**kwargs_train))
        val_dataset_full   = RetMapsDataset(data_dir, "val", transform=MapTransform(**kwargs_val))
        
        # Use the same indices split for both datasets.
        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset   = Subset(val_dataset_full, val_indices)
        
        # --- Create DataLoaders ---
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn)
        
        print("DataLoader initialized.")
        return train_loader, val_loader
    
    elif data_split == "train+val":
        print("Using both training and validation datasets.")
        # --- Training dataset from the "train" folder ---
        kwargs_train = kwargs.copy()
        kwargs_train["use_aug"] = kwargs.get("use_aug", True)
        train_dataset = RetMapsDataset(data_dir, "train", transform=MapTransform(**kwargs_train))
        
        # --- Validation dataset from the "val" folder (always no augmentation) ---
        kwargs_val = kwargs.copy()
        kwargs_val["use_aug"] = False
        val_dataset = RetMapsDataset(data_dir, "val", transform=MapTransform(**kwargs_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn)
        
        print("DataLoader initialized.")
        return train_loader, val_loader
    
    else:
        raise ValueError(f"Invalid data_split: {data_split}. Use 'val' or 'train+val'.")
    
    
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