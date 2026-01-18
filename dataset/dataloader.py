# Library imports
import json
import os
import random
from collections.abc import Sequence

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add base path to PYTHONPATH
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)
    
# Local imports
from dataset.load_episodes import load_episodes, convert_floor_ep
from dataset.maps.base_map import HabtoGrid
from dataset.transform import MapTransform

with open("data/val/maps/id_to_floor.json", "r") as f:
    ID_TO_FLOOR_VAL = json.load(f)

class RetMapsNewDataset(Dataset):
    """
    Dataset variant that works with the HM3D difficulty-based splits produced by
    the updated load_episodes helper.
    """

    map_base_dir = "data/val/maps"
    map = HabtoGrid(embeds_dir=map_base_dir)

    def __init__(
        self,
        difficulty: str = "easy",
        episodes_base_dir: str = "data/val",
        split_dir: str = "splits",
        episodes=None,
        transform=None,
    ):
        if episodes is None:
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
        
        if isinstance(query, (list, tuple)) and query:
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


def _split_episodes(episodes: list, val_ratio: float, seed: int):
    if len(episodes) < 2:
        raise ValueError("Need at least 2 episodes to split into train/val.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    indices = list(range(len(episodes)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = int(len(indices) * val_ratio)
    val_count = max(1, min(len(indices) - 1, val_count))

    val_indices = set(indices[:val_count])
    train_episodes = [episodes[i] for i in indices if i not in val_indices]
    val_episodes = [episodes[i] for i in indices if i in val_indices]

    return train_episodes, val_episodes


def get_dataloaders_new_split(
    levels: Sequence[str],
    episodes_base_dir: str = "data/val",
    split_dir: str = "splits",
    batch_size: int = 32,
    num_workers: int = 4,
    collate_fn=None,
    augmentation=None,
    val_ratio: float = 0.2,
    seed: int = 2025,
):
    """
    Creates train/val DataLoaders by merging difficulty levels and splitting episodes.
    """
    levels_list = [levels] if isinstance(levels, str) else list(levels)
    levels_label = ",".join(levels_list)
    print(f"Initializing HM3D train/val DataLoaders for '{levels_label}'...")

    episodes = []
    for level in levels_list:
        episodes.extend(
            load_episodes(
                base_dir=episodes_base_dir,
                split_dir=split_dir,
                split=level,
            )
        )

    train_episodes, val_episodes = _split_episodes(episodes, val_ratio, seed)

    train_transform = None
    if augmentation and augmentation.get("use_aug"):
        train_transform = MapTransform(augmentation)

    train_dataset = RetMapsNewDataset(
        episodes=train_episodes,
        episodes_base_dir=episodes_base_dir,
        split_dir=split_dir,
        transform=train_transform,
    )
    val_dataset = RetMapsNewDataset(
        episodes=val_episodes,
        episodes_base_dir=episodes_base_dir,
        split_dir=split_dir,
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print("HM3D train/val DataLoaders initialized.")
    return train_loader, val_loader


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
