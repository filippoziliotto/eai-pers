
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
        if entry["floor_id"] == floor_id:
            return entry["episode_id"]
    return None

def load_episodes(
    base_dir: str = "data/v2/splits",
    split_dir: str = "objects_unseen", 
    split: str = "train",
    ) -> List[Dict]:
    """
    Reads all 'episodes.json' files from subdirectories in base_dir and merges them into a single list.

    Parameters:
        base_dir (str): The base directory containing subfolders with 'episodes.json' files.
        split_dir (str): The subdirectory containing the split data. (default: "objects_unseen")
        split (str): The specific split to load (e.g., "train", "val"). (default: "train")

    Returns:
        list[dict]: A combined list of all episodes from all subfolders.
    """
    episodes = []
    base_dir = os.path.join(base_dir, split_dir, split)

    # Traverse through each subdirectory
    for root, _, files in os.walk(base_dir):
        # Check if 'episodes.json' exists in the current directory
        assert os.path.exists(root), f"Directory {root} does not exist!"
        assert "episodes.json" in files, f"episodes.json not found in {root}!"
        
        # Read the 'episodes.json' file
        file_path = os.path.join(root, "episodes.json")
        with open(file_path, "r") as f:
            ep = json.load(f)
            episodes.extend(ep)  # Append all episodes to the list
    
    assert len(episodes) > 0, "No episodes found!"
    return episodes


if __name__ == "__main__":
    base_path = "data/v2/splits"
    episodes_dir = os.path.join(base_path, "objects_unseen")
    split = "train"

    episodes = load_episodes(base_path, episodes_dir, split)
    
    
    
