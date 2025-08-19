
# Library imports
import os
import json
import gzip
from typing import List, Dict

# Load episode_id to floor_id mapping
with open("data/v2/maps/id_to_floor.json", "r") as f:
    ID_TO_FLOOR = json.load(f)
    
def convert_floor_ep(data, scene_name, floor_id):
    """
    Given the data, scene_name, and floor_id, return the corresponding episode_id.
    """
    for entry in data.get(scene_name, []):
        if entry["floor_id"] == int(floor_id):
            return entry["episode_id"]
    
    assert False, f"Episode not found for scene {scene_name} and floor_id {floor_id}"


def load_episodes(
    base_dir: str = "../data/v2/splits",
    split_dir: str = "objects_unseen", 
    split: str = "train",
    val_subsplit: str = "easy",
    ) -> List[Dict]:
    """
    Reads all .json files from base_dir/split_dir/split and merges them into a single list.
    - base_dir/split_dir/train/<scene>/episodes.json  (train)
    - base_dir/split_dir/val/easy/content/*.json  (val easy/medium/hard)
    """
    episodes = []
    full_base_path = os.path.join(base_dir, "splits", split_dir, split)
   
    # Remove invalid scenes
    invalid_scenes = ["k1cupFYWXJ6", "HY1NcmCgn3n", "7MXmsvcQjpJ"]

    if split == "train":
        # Old structure: subfolders for each scene
        # List all immediate subdirectories
        for subdir in os.listdir(full_base_path):
            subdir_path = os.path.join(full_base_path, subdir)

            # Skip invalid scenes
            if subdir in invalid_scenes:
                continue

            if os.path.isdir(subdir_path):
                episode_file = os.path.join(subdir_path, "episodes.json")
                assert os.path.exists(episode_file), f"'episodes.json' not found in {subdir_path}"

                with open(episode_file, "r") as f:
                    ep = json.load(f)
                    episodes.extend(ep)
    
    elif split == "val":

        if val_subsplit == "total":
            # Merge all files easy, medium, hard
            for difficulty in ["easy", "medium", "hard"]:
                difficulty_path = os.path.join(full_base_path, difficulty, "content/",)
                
                for fname in os.listdir(difficulty_path):
                    if fname.endswith(".json.gz"):
                        file_path = os.path.join(difficulty_path, fname)
                        
                        with gzip.open(file_path, "rt", encoding="utf-8") as f:
                            data = json.load(f)
                            episodes.append(data)
        else:
            # All files .json in easy/medium/hard folder
            split_path = os.path.join(full_base_path, val_subsplit, "content/")
            for fname in os.listdir(split_path):
                if fname.endswith(".json.gz"):
                    file_path = os.path.join(split_path, fname)
               
                    with gzip.open(file_path, "rt", encoding="utf-8") as f:
                        ep = json.load(f)
                        episodes.append(ep)

   
    assert len(episodes) > 0, "No episodes found!"
    return episodes


if __name__ == "__main__":
    base_path = "data/v2/splits"
    episodes_dir = os.path.join(base_path, "objects_unseen")
    split = "train"

    episodes = load_episodes(base_path, episodes_dir, split)
    
    
    
