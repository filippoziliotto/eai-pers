
# Library imports
import os
import json
import gzip
from typing import Dict, List

# Load episode_id to floor_id mapping (kept for compatibility with downstream code)
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


def _load_scene_file(file_path: str) -> Dict:
    """
    Loads a single content file, handling both JSON and JSON.GZ extensions.
    """
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_content_files(content_dir: str) -> List[str]:
    """
    Returns a deterministic list of paths to load. When both .json and .json.gz
    versions exist for the same scene, prefer the uncompressed file.
    """
    file_map: Dict[str, str] = {}
    for name in os.listdir(content_dir):
        if not (name.endswith(".json") or name.endswith(".json.gz")):
            continue
        base_name = name[:-3] if name.endswith(".gz") else name
        resolved = os.path.join(content_dir, name)
        # Prefer plain JSON over JSON.GZ when both exist.
        if base_name not in file_map or file_map[base_name].endswith(".gz"):
            file_map[base_name] = resolved
    return sorted(file_map.values())


def load_episodes(
    base_dir: str = "data/val",
    split_dir: str = "splits",
    split: str = "easy",
) -> List[Dict]:
    """
    Loads and merges HM3D episodes for a given difficulty split.

    Args:
        base_dir: Root directory that contains the validation data.
        split_dir: Intermediate directory that stores split folders (default: ``splits``).
        split: Difficulty name (e.g., ``easy``, ``medium``, ``hard``).
    """
    content_dir = os.path.join(base_dir, split_dir, split, "content")
    assert os.path.isdir(content_dir), f"Content directory not found: {content_dir}"

    episodes: List[Dict] = []
    for file_path in _resolve_content_files(content_dir):
        scene_content = _load_scene_file(file_path)
        episodes.extend(active_to_passive_ep(scene_content))

    assert episodes, f"No episodes found for difficulty '{split}'."
    return episodes


def active_to_passive_ep(scene_content: Dict) -> List[Dict]:
    """
    For a single scene file, merges episode information with the corresponding
    goal metadata (minus the view points) and returns the flattened episodes.
    """
    episodes = scene_content.get("episodes", [])
    goals_by_category = scene_content.get("goals_by_category", {})

    goal_by_name: Dict[str, Dict] = {}
    goal_by_id: Dict[int, Dict] = {}

    for goals in goals_by_category.values():
        if not isinstance(goals, list):
            continue
        for goal in goals:
            if not isinstance(goal, dict):
                continue
            goal_clean = {k: v for k, v in goal.items() if k != "view_points"}
            object_name = goal_clean.get("object_name")
            if object_name:
                goal_by_name[object_name] = goal_clean
            object_id = goal_clean.get("object_id")
            if isinstance(object_id, int):
                goal_by_id[object_id] = goal_clean

    merged_episodes: List[Dict] = []
    for episode in episodes:
        merged_episode = episode.copy()
        goal_metadata = None

        object_name = merged_episode.get("object_id") or merged_episode.get("object_name")
        if isinstance(object_name, str):
            goal_metadata = goal_by_name.get(object_name)
            if goal_metadata is None:
                suffix = object_name.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    goal_metadata = goal_by_id.get(int(suffix))

        if goal_metadata is None:
            closest_goal_id = merged_episode.get("info", {}).get("closest_goal_object_id")
            if isinstance(closest_goal_id, int):
                goal_metadata = goal_by_id.get(closest_goal_id)

        if goal_metadata:
            merged_episode.update(goal_metadata)
            if "position" in goal_metadata:
                merged_episode.setdefault("object_pos", goal_metadata["position"])
        else:
            # If we cannot match a goal entry, still keep the episode as-is.
            merged_episode.setdefault("object_pos", merged_episode.get("position"))

        merged_episodes.append(merged_episode)

    return merged_episodes


if __name__ == "__main__":
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes.")
