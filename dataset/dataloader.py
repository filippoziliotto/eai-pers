# Library imports
import json
import os
from collections.abc import Sequence

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add base path to PYTHONPATH
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)
    
# Local imports
from dataset.load_episodes import load_episodes, convert_floor_ep, ID_TO_FLOOR
from dataset.maps.base_map import HabtoGrid
from dataset.transform import MapTransform
from dataset.naming import NameSelector
import random

with open("data/val/maps/id_to_floor.json", "r") as f:
    ID_TO_FLOOR_VAL = json.load(f)

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
        if isinstance(self.selector.names, set):
            self.selector.names = sorted(self.selector.names)
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
        
        # Save Visualizations before and after transformations
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
                                  num_workers=num_workers, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn, drop_last=False)
        
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
                                  num_workers=num_workers, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn, drop_last=False)
        
        print("DataLoader initialized.")
        return train_loader, val_loader
    
    
#######################
# WRITE HERE NEW CODE #
#######################


class RetMapsNewDataset(Dataset):
    """
    Dataset variant that works with the HM3D difficulty-based splits produced by
    the updated load_episodes helper.
    """

    map_base_dir = "data/val/maps"
    map = HabtoGrid(embeds_dir=map_base_dir)
    selector = NameSelector()

    def __init__(
        self,
        difficulty: str = "easy",
        episodes_base_dir: str = "data/val",
        split_dir: str = "splits",
        transform=None,
    ):
        if isinstance(difficulty, str):
            difficulties = [difficulty]
        elif isinstance(difficulty, Sequence):
            difficulties = list(difficulty)
        else:
            difficulties = [difficulty]

        episodes = []
        for level in difficulties:
            episodes.extend(
                load_episodes(
                    base_dir=episodes_base_dir,
                    split_dir=split_dir,
                    split=level,
                )
            )

        self.episodes = episodes
        self.transform = transform

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int):
        episode = self.episodes[idx]
        scene_name = episode["scene_id"].split("/")[-1].split(".")[0]

        ext_summary = episode["extracted_summary"]
        summary = episode["summary"]
        query = episode["query"]
        floor_id = episode["floor_id"]
        target_pos_hab = np.array([episode["object_pos"]])

        self.map.load_embed_init(
            scene_name=scene_name,
            base_dir=self.map_base_dir,
            episode_id=convert_floor_ep(ID_TO_FLOOR_VAL, scene_name, floor_id),
        )

        feature_map = self.map.load_embed_np_arr(visualize=False)
        feature_map = torch.from_numpy(feature_map)

        target = self.map.hab_to_px(target_pos_hab[:, [0, 2]])
        target = self.map.px_to_arr(
            target,
            (self.map.init_dict["map_shape"] // self.map.grid_size) // 2,
        )[0]

        if self.transform:
            feature_map, target, ext_summary = self.transform(
                feature_map, target, ext_summary
            )
            
        # Visualizations after Map Transform (useful to check if trasnformation is ok)
        #self.map.visualize(
        #    arr=feature_map,
        #    target=target,
        #    save_to_disk=True,
        #    path_to_image="path_tofolder/posttransform.png"
        #)
        
        # Tke a random query
        query = random.choice(query)

        return {
            "scene_name": scene_name,
            "floor_id": floor_id,
            "t_summary": summary,
            "summary": ext_summary,
            "target": target,
            "query": query,
            "feature_map": feature_map,
        }


def get_dataloader_new(
    difficulty: str = "easy",
    episodes_base_dir: str = "data/val",
    split_dir: str = "splits",
    batch_size: int = 32,
    num_workers: int = 4,
    collate_fn=None,
    augmentation=None,
    shuffle: bool = False,
):
    """
    Creates a DataLoader backed by ``RetMapsNewDataset`` for the HM3D difficulty splits.
    ``difficulty`` can be a single level or a list/tuple of levels to merge.
    """
    if isinstance(difficulty, Sequence) and not isinstance(difficulty, str):
        difficulty_label = ",".join([str(level) for level in difficulty])
    else:
        difficulty_label = str(difficulty)
    print(f"Initializing HM3D DataLoader for difficulty '{difficulty_label}'...")

    transform = None
    if augmentation and augmentation.get("use_aug"):
        transform = MapTransform(augmentation)

    dataset = RetMapsNewDataset(
        difficulty=difficulty,
        episodes_base_dir=episodes_base_dir,
        split_dir=split_dir,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print("HM3D DataLoader initialized.")
    return dataloader


def main():
    """
    Quick sanity script that instantiates the HM3D dataloader and prints one batch.
    """
    loader = get_dataloader_new(
        difficulty="easy",
        batch_size=2,
        num_workers=0,
        augmentation=None,
        shuffle=False,
    )
    batch = next(iter(loader))

    print("Batch keys:", list(batch.keys()))
    feature_map = batch["feature_map"]
    print("Feature map batch shape:", feature_map.shape)

    print("Scene names:", batch["scene_name"])
    print("Floor IDs:", batch["floor_id"])
    print("Targets tensor shape:", batch["target"].shape)
    print("Sample query:", batch["query"][0] if batch["query"] else None)


if __name__ == "__main__":
    main()
