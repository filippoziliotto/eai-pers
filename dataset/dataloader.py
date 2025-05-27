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
from dataset.load_episodes import load_episodes, convert_floor_ep, ID_TO_FLOOR
from dataset.maps.base_map import HabtoGrid
from dataset.transform import MapTransform
from dataset.naming import NameSelector

class RetMapsDataset(Dataset):
    """
    Dataset class for retrieving maps and their corresponding descriptions.
    Load everything in memory to train/eval
    """
    # Load map class
    base_dir = "data/v2/maps"
    map = HabtoGrid(embeds_dir = base_dir)
    selector = NameSelector()
    
    def __init__(self, data_dir="data/v2/", split_dir="object_unseen", data_split="train", transform=None):
        
        self.episodes = load_episodes(data_dir, split_dir, data_split)        
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

        # Apply realistic names to the episode
        episode = self.selector.apply_names(episode)

        # Extract episode information
        ext_summary = episode["extracted_summary"]
        summary = episode["summary"]
        query = episode["query"]
        floor_id = episode["floor_id"]
        target_pos_hab = np.array([episode["object_pos"]])

        # Load the corresponding map embeddings
        self.map.load_embed_init(
            scene_name=scene_name,
            base_dir=self.base_dir,
            episode_id=convert_floor_ep(ID_TO_FLOOR, scene_name, floor_id),
        )

        # Load the feature map for the current episode
        feature_map = self.map.load_embed_np_arr(visualize=False)
        feature_map = torch.from_numpy(feature_map)
        
        # TODO: here we create the graph

        # Convert the target position from habitat coordinates to map frame
        target = self.map.hab_to_px(target_pos_hab[:, [0, 2]])
        target = self.map.px_to_arr(
            target,
            (self.map.init_dict['map_shape'] // self.map.grid_size) // 2
        )[0]
        
        # Save feature map. sum() + target in an image
        # save it to trainer/visualizations/pretransform.png
        #self.map.visualize(
        #    arr=feature_map,
        #    target=target,
        #    save_to_disk=True,
        #    path_to_image="trainer/visualizations/pretransform.png"
        #)
        
        # Apply optional transformations (e.g., tensor conversion, augmentations)
        if self.transform:
            feature_map, target, ext_summary = self.transform(feature_map, target, ext_summary)

        #self.map.visualize(
        #    arr=feature_map,
        #    target=target,
        #    save_to_disk=True,
        #    path_to_image="trainer/visualizations/posttransform.png"
        #)

        # Package and return the sample as a dictionary
        return {
            "scene_name": scene_name,
            "floor_id": floor_id,
            "t_summary": summary,
            "summary": ext_summary,
            "target": target,
            "query": query,
            "feature_map": feature_map,
        }

def get_dataloader(data_dir, 
                   split_dir="object_unseen", 
                   batch_size=32, 
                   num_workers=4, 
                   collate_fn=None,
                   augmentation=None,):
    """
    Returns DataLoaders based on the specified data_split.
    
    Modes:
      - "object_unseen": #TODO:
      - "scene_unseen": #TODO:
      
    Parameters:
      data_dir (str): Path to the dataset.
      split_dir (str): Split directory (e.g., "object_unseen", "scene_unseen").
      batch_size (int): Batch size.
      num_workers (int): Number of DataLoader workers.
      collate_fn (callable): Custom collate function.
      augmentation (dict): Augmentation configuration.      
    Returns:
      Tuple (train_loader, val_loader)
    """
    print("Initializing DataLoader...")
    
    if split_dir == "object_unseen":
        print("Setting: Object Unseen")
        print("Using both training and validation datasets.")
        # --- Training dataset from the "train" folder ---
        aug_train = MapTransform(augmentation) if augmentation["use_aug"] else None
        train_dataset = RetMapsDataset(data_dir, split_dir, "train" , transform=aug_train)
        
        # --- Validation dataset from the "val" folder (always no augmentation) ---
        val_dataset = RetMapsDataset(data_dir, split_dir, "val", transform=None)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn)
        
        print("DataLoader initialized.")
        return train_loader, val_loader
    
    elif split_dir == "scene_unseen":
        print("Setting: Scene Unseen")
        print("Using both training and validation datasets.")
        # --- Training dataset from the "train" folder ---
        aug_train = MapTransform(augmentation) if augmentation["use_aug"] else None
        train_dataset = RetMapsDataset(data_dir, split_dir, "train", transform=MapTransform(aug_train))
        
        # --- Validation dataset from the "val" folder (always no augmentation) ---
        val_dataset = RetMapsDataset(data_dir, split_dir, "val", transform=None)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn)
        
        print("DataLoader initialized.")
        return train_loader, val_loader