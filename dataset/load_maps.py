import os
import json
from typing import List, Dict

def load_all_episodes(base_dir: str, split:str) -> List[Dict]:
    """
    Reads all 'episodes.json' files from subdirectories in base_dir and merges them into a single list.

    Parameters:
        base_dir (str): The base directory containing subfolders with 'episodes.json' files.

    Returns:
        list[dict]: A combined list of all episodes from all subfolders.
    """
    all_episodes = []
    
    base_dir = os.path.join(base_dir, split)

    # Traverse through each subdirectory
    for root, _, files in os.walk(base_dir):
        if "episodes.json" in files:
            file_path = os.path.join(root, "episodes.json")
            with open(file_path, "r") as f:
                episodes = json.load(f)
                all_episodes.extend(episodes)  # Append all episodes to the list

    return all_episodes

def load_all_maps(maps_base_dir: str) -> List[Dict]:
    """
    Reads 'pos_vars.json' and 'feature_map.npz' from each subfolder in maps_base_dir 
    and extracts scene, robot data, and feature map path.

    Parameters:
        maps_base_dir (str): The base directory containing subfolders named '<scene_id>_floor_<floor_id>'.

    Returns:
        list[dict]: A list where each dictionary represents a map subfolder with extracted data.
    """
    all_maps = []

    # Traverse through each subdirectory
    for subfolder in os.listdir(maps_base_dir):
        subfolder_path = os.path.join(maps_base_dir, subfolder)

        # Ensure it's a directory and follows the expected naming pattern
        if not os.path.isdir(subfolder_path) or "_floor_" not in subfolder:
            continue

        # Extract scene_id and floor_id
        scene_id, floor_str = subfolder.split("_floor_")
        try:
            floor_id = int(floor_str)  # Convert to integer
        except ValueError:
            continue  # Skip if floor_id is not a valid integer

        # Paths for files
        pos_vars_path = os.path.join(subfolder_path, "pos_vars.json")
        feature_map_path = os.path.join(subfolder_path, "feature_map.npz")

        if os.path.exists(pos_vars_path):
            # Load pos_vars.json
            with open(pos_vars_path, "r") as f:
                pos_vars = json.load(f)

            if isinstance(pos_vars, list) and len(pos_vars) > 0:
                first_entry = pos_vars[0]  # Take only the first dictionary

                # Construct dictionary
                map_data = {
                    "scene_id": scene_id,
                    "floor_id": floor_id,
                    "robot_xyz": first_entry.get("start_position", None),
                    "robot_xy": first_entry.get("robot_xy", None),
                    "robot_heading": first_entry.get("heading", None),
                    "robot_rotation": first_entry.get("start_rotation", None),
                    "feature_map_path": feature_map_path if os.path.exists(feature_map_path) else None,
                }

                # Append to list
                all_maps.append(map_data)

    return all_maps

def filter_episodes_by_maps(
    all_episodes: List[Dict], all_maps: List[Dict], split:str
) -> List[Dict]:
    """
    Filters all_episodes to retain only episodes with matching scene_id and floor_id in all_maps.
    Adds robot_xyz, robot_xy, robot_heading, and feature_map_path to the episode dictionary.

    Parameters:
        all_episodes (list[dict]): List of episode dictionaries.
        all_maps (list[dict]): List of map dictionaries.

    Returns:
        list[dict]: Filtered and updated list of episodes.
    """
    # Create a map for fast lookup: (scene_id, floor_id) -> map_data
    map_lookup = {
        (m["scene_id"], m["floor_id"]): m for m in all_maps
    }

    filtered_episodes = []

    # Loop through all episodes and filter them
    for ep in all_episodes:
        scene_id = ep["scene_id"].split("/")[-1].split('.')[0]  # Extract scene_id from ep["scene_id"]

        # Check if there's a matching map
        if (scene_id, ep["floor_id"]) in map_lookup:
            # Get the corresponding map data
            map_data = map_lookup[(scene_id, ep["floor_id"])]

            # Add keys to the episode
            ep["robot_xyz"] = map_data.get("robot_xyz", None)
            ep["robot_xy"] = map_data.get("robot_xy", None)
            ep["robot_heading"] = map_data.get("robot_heading", None)
            ep["robot_rot"] = map_data.get("robot_rotation", None)
            ep["feature_map_path"] = map_data.get("feature_map_path", None)

            # Add the updated episode to the filtered list
            filtered_episodes.append(ep)
            filtered_episodes[-1]["episode_id"] = len(filtered_episodes)-1

    print(f"N° {split} Episodes: {len(filtered_episodes)}")
    return filtered_episodes

def load_episodes(base_path:str, split:str) -> List[Dict]:
    """
    Load episodes from the episodes directory and filter them based on available maps.

    Parameters:
        episodes_dir (str): Path to the episodes directory.
        maps_dir (str): Path to the maps directory.

    Returns:
        list[dict]: Filtered and updated list of episodes.
    """
    episodes_dir = os.path.join(base_path, split, "episodes")
    maps_dir = os.path.join(base_path, split, "maps")
    
    all_episodes = load_all_episodes(episodes_dir, split)
    all_maps = load_all_maps(maps_dir)
    filtered_episodes = filter_episodes_by_maps(all_episodes, all_maps, split)
    
    assert len(all_episodes) > 0, "No episodes found!"
    assert len(all_maps) > 0, "No maps found!"
    assert len(filtered_episodes) > 0, "No episodes with matching maps found"

    return filtered_episodes


# We extracted the episodes descriptions with GPT and saved the JSON with the extracted summaries.
# TODO: This is messy we need to adjust this
def load_extracted_episodes(base_path: str, split: str) -> List[Dict]:
    """
    Load the extracted episodes from the JSON file.

    Returns:
        list[dict]: List of episodes with extracted summaries.
    """
    extracted_file = os.path.join(base_path, split, "filtered_episodes.json")
    #extracted_file = "data/val/filtered_episodes.json"
    with open(extracted_file, "r") as f:
        extracted_episodes = json.load(f)
    return extracted_episodes

def create_valid_scenes_episodes(base_path: str, split: str, save_to_json: bool = False) -> Dict[str, List[str]]:
    """
    Load and group map subfolders by scene name and optionally save the result to a JSON file.

    Parameters:
        base_path (str): The base directory containing the dataset.
        split (str): The data split (e.g., 'train', 'val').
        save_to_json (bool): If True, saves the grouped scenes to a JSON file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping each scene name to a list of floor subfolders.
    """
    maps_dir = os.path.join(base_path, split, "maps")
    
    # Get all valid scenes folders where the folder name contains "_floor_"
    scenes = [
        folder for folder in os.listdir(maps_dir)
        if os.path.isdir(os.path.join(maps_dir, folder)) and "_floor_" in folder
    ]
    
    # Group folders by scene name (extracted from the part before "_floor_")
    valid_scenes = {}
    for folder in scenes:
        scene_name = folder.split("_floor_")[0]
        valid_scenes.setdefault(scene_name, []).append(folder)

    # Optionally save the valid_scenes dictionary to a JSON file
    if save_to_json:
        episodes_dir = os.path.join(base_path, split, "episodes", split)
        os.makedirs(episodes_dir, exist_ok=True)
        json_path = os.path.join(episodes_dir, "valid_scenes_and_floors.json")
        with open(json_path, "w") as f:
            json.dump(valid_scenes, f, indent=4)

    return valid_scenes
    

if __name__ == "__main__":
    base_path = "data"
    split = "train"
    episodes = load_episodes(base_path, split)
    print(episodes[:2])  # Display first two episodes
    
    # This has to be done first to create the valid_scenes episodes
    #valid_scenes = create_valid_scenes_episodes(base_path, split, save_to_json=False)
    
    
    
