
# Library imports
import os
import json
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
    base_dir: str = "data/v2/splits",
    split_dir: str = "objects_unseen", 
    split: str = "train",
    ) -> List[Dict]:
    """
    Reads all 'episodes.json' files from subdirectories of base_dir/split_dir/split 
    and merges them into a single list.
    """
    episodes = []
    full_base_path = os.path.join(base_dir, "v2/", "splits", split_dir, split)
    
    # Remove invalid scenes
    invalid_scenes = ["k1cupFYWXJ6", "HY1NcmCgn3n", "7MXmsvcQjpJ"]

    # List all immediate subdirectories
    for subdir in os.listdir(full_base_path):
        print(subdir)
        subdir_path = os.path.join(full_base_path, subdir)
        
        # Skip invalid scenes
        if subdir in invalid_scenes:
            continue
        
        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            episode_file = os.path.join(subdir_path, "episodes.json")
            assert os.path.exists(episode_file), f"'episodes.json' not found in {subdir_path}"
            
            with open(episode_file, "r") as f:
                ep = json.load(f)
                episodes.extend(ep)

    assert len(episodes) > 0, "No episodes found!"
    return episodes

if __name__ == "__main__":
    base_path = "data/v2/splits"
    episodes_dir = os.path.join(base_path, "objects_unseen")
    split = "train"

    episodes = load_episodes(base_path, episodes_dir, split)
    
    
    
