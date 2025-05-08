# Library imports
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import os
import json

# Add base path to PYTHONPATH
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)
    
# Local imports
from dataset.load_episodes import load_episodes, convert_floor_ep, ID_TO_FLOOR
from dataset.maps.base_map import HabtoGrid
from dataset.transform import MapTransform

# Config
import config

class RetMapsDataset(Dataset):
    """
    Dataset class for retrieving maps and their corresponding descriptions.
    Load everything in memory to train/eval
    """
    # Load map class
    map = HabtoGrid(embeds_dir = "data/v2/maps")
    base_dir = "data/v2/maps"

    def __init__(self, data_dir="data", data_split="train", transform=None):
        
        self.episodes = load_episodes(data_dir, data_split)
        self.episodes_dir = os.path.join(data_dir, data_split)
        
        self.transform = transform

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single episode sample.

        Args:
            idx (int): Index of the episode.

        Returns:
            dict: Dictionary containing processed feature map, target, query, and description.
        """
        # Retrieve episode data
        episode = self.episodes[idx]
        scene_name = episode["scene_id"].split("/")[-1].split(".")[0]

        # Extract episode information
        ext_summary = episode["extracted_summary"]
        # TODO: add possibility to load only the summary
        summary = episode["summary"]
        query = episode["query"]
        floor_id = episode["floor_id"]
        target_pos_hab = np.array(episode["position"])

        # Load the corresponding map embeddings
        self.map.load_embed_init(
            scene_name=scene_name,
            base_dir=self.base_dir,
            episode_id=convert_floor_ep(ID_TO_FLOOR, scene_name, floor_id),
        )

        # Load the feature map for the current episode
        feature_map = self.map.load_embed_np_arr(visualize=False)

        # Convert the target position from habitat coordinates to map frame
        target = self.map.hab_to_px(target_pos_hab[:, [0, 2]])
        target = self.map.px_to_arr(
            target,
            (self.map.init_dict['map_shape'] // self.map.grid_size) // 2
        )

        # Apply optional transformations (e.g., tensor conversion, augmentations)
        if self.transform:
            feature_map, target, ext_summary = self.transform(feature_map, target, ext_summary)

        # Package and return the sample as a dictionary
        return {
            "summary": ext_summary,
            "target": target,
            "query": query,
            "feature_map": feature_map,
        }

def get_dataloader(data_dir, 
                   data_split="object_unseen", 
                   batch_size=32, 
                   num_workers=4, 
                   collate_fn=None,
                   kwargs={}):
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
      kwargs (dict): Additional keyword arguments for MapTransform.
      
    Returns:
      Tuple (train_loader, val_loader)
    """
    print("Initializing DataLoader...")
    
    if data_split == "object_unseen":
        print("Setting: Object Unseen")
        print("Using both training and validation datasets.")
        # --- Training dataset from the "train" folder ---
        kwargs_train = kwargs.copy()
        kwargs_train["use_aug"] = kwargs.get("use_aug", True)
        train_dataset = RetMapsDataset(data_dir, "train", transform=MapTransform(**kwargs_train))
        
        # --- Validation dataset from the "val" folder (always no augmentation) ---
        kwargs_val = kwargs.copy()
        kwargs_val["use_aug"] = False
        val_dataset = RetMapsDataset(data_dir, "val", False, transform=MapTransform(**kwargs_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn)
        
        print("DataLoader initialized.")
        return train_loader, val_loader
    
    elif data_split == "scene_unseen":
        print("Setting: Scene Unseen")
        print("Using both training and validation datasets.")
        # --- Training dataset from the "train" folder ---
        kwargs_train = kwargs.copy()
        kwargs_train["use_aug"] = kwargs.get("use_aug", True)
        train_dataset = RetMapsDataset(data_dir, "train", transform=MapTransform(**kwargs_train))
        
        # --- Validation dataset from the "val" folder (always no augmentation) ---
        kwargs_val = kwargs.copy()
        kwargs_val["use_aug"] = False
        val_dataset = RetMapsDataset(data_dir, "val", False, transform=MapTransform(**kwargs_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn)
        
        print("DataLoader initialized.")
        return train_loader, val_loader
    

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