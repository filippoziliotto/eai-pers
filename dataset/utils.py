import math
import re
from copy import deepcopy
from typing import Any, List, Set, Dict, Union, Sequence

# Torch imports
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


"""
BATCHING UTILITIES
"""

def custom_collate(batch):
    """
    Custom collate function that pads feature maps (which may have been randomly cropped)
    to a common size before stacking them into a batch.
    
    Each feature map is assumed to have shape (H, W, C) (i.e. height, width, channels).
    The padding is applied to the bottom and right so that each feature map reaches the
    maximum height and width found in the batch.
    """
    # Rimuovi gli item None
    batch = [item for item in batch if item is not None]

    if not batch:
        # Puoi anche restituire un batch "vuoto" gestibile se preferisci
        return None

    
    # Extract descriptions, queries, targets, and map paths.
    summaries = [item["summary"] for item in batch]
    queries = [item["query"] for item in batch]
    targets = torch.stack([torch.tensor(item["target"]) for item in batch])

    # Get the list of feature maps from the batch.
    feature_maps = [item["feature_map"] for item in batch]

    # Check if the feature maps are already padded.
    if not all(fm.shape[:2] == feature_maps[0].shape[:2] for fm in feature_maps):
        # Determine the maximum height and width among all feature maps.
        max_h = max(fm.shape[0] for fm in feature_maps)
        max_w = max(fm.shape[1] for fm in feature_maps)

        padded_feature_maps = []
        for fm in feature_maps:
            h, w, c = fm.shape  # assuming fm has shape (H, W, C)
            # Compute how much to pad on the bottom and right.
            pad_bottom = max_h - h
            pad_right = max_w - w

            # torch.nn.functional.pad expects the padding configuration for the last two dims
            # in the order (pad_left, pad_right, pad_top, pad_bottom). Because our feature map
            # is (H, W, C) (with channels last), we first permute it to (C, H, W) before padding.
            fm_perm = fm.permute(2, 0, 1)  # now shape is (C, H, W)
            # Apply padding: we leave the top and left untouched, so pad_left and pad_top are 0.
            fm_padded = F.pad(fm_perm, (0, pad_right, 0, pad_bottom))
            # Permute back to (H, W, C)
            fm_padded = fm_padded.permute(1, 2, 0)
            padded_feature_maps.append(fm_padded)

        # Now stack the padded feature maps to form a single tensor.
        feature_maps = torch.stack(padded_feature_maps)
        
    else:
        # If the feature maps are already padded, stack them as they are.
        feature_maps = torch.stack(feature_maps)

    return {
        "summary": summaries,
        "query": queries,
        "target": targets,
        "feature_map": feature_maps,
    }
    
"""
Transforms Utils for Maps
"""

def random_crop_preserving_target(
    feature_map: torch.Tensor,                    # (H, W, E)
    xy_coords: Union[Sequence[float], torch.Tensor],  # [y, x], floats or tensor
    max_crop_fraction: float = 0.3,
    min_size: tuple[int, int] = (20, 20)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly crop `feature_map`, ensuring the point `xy_coords` remains inside the crop,
    and adjust `xy_coords` to the new crop coordinate frame.

    Args:
        feature_map: Tensor of shape (H, W, E).
        xy_coords: Sequence or Tensor of shape (2,), floats [y, x].
        max_crop_fraction: Maximum fraction of H,W you may remove.
        min_size: Minimum (height, width) of the crop.

    Returns:
        cropped_map: Tensor of shape (new_h, new_w, E).
        new_xy: Tensor of shape (2,), floats [y', x'] in the cropped map.
    """
    # Ensure xy_coords is a float tensor on the same device
    if not torch.is_tensor(xy_coords):
        xy = torch.tensor(xy_coords, dtype=torch.float32, device=feature_map.device)
    else:
        xy = xy_coords.to(device=feature_map.device, dtype=torch.float32)
    y, x = float(xy[0]), float(xy[1])

    H, W, _ = feature_map.shape

    # If the map is already smaller than min_size, skip cropping
    if H <= min_size[0] or W <= min_size[1]:
        return feature_map, xy.clone()

    # Compute max removal in pixels
    max_crop_h = int(math.floor(H * max_crop_fraction))
    max_crop_w = int(math.floor(W * max_crop_fraction))

    # Compute allowed new sizes
    min_h, min_w = min_size
    lower_h = max(min_h, H - max_crop_h)
    lower_w = max(min_w, W - max_crop_w)

    # Sample new crop size uniformly in [lower, original]
    new_h = torch.randint(lower_h, H + 1, (1,)).item()
    new_w = torch.randint(lower_w, W + 1, (1,)).item()

    # Determine valid top/left so (y,x) stays inside
    # top ≤ y ≤ top + new_h - 1  ⇒  top ∈ [y - (new_h - 1), y]
    top_min = max(0, int(math.ceil(y - (new_h - 1))))
    top_max = min(int(math.floor(y)), H - new_h)
    left_min = max(0, int(math.ceil(x - (new_w - 1))))
    left_max = min(int(math.floor(x)), W - new_w)

    # If invalid, skip cropping
    if top_min > top_max or left_min > left_max:
        return feature_map, xy.clone()

    top = torch.randint(top_min, top_max + 1, (1,)).item()
    left = torch.randint(left_min, left_max + 1, (1,)).item()

    # Crop and adjust coords
    cropped = feature_map[top: top + new_h, left: left + new_w, :]
    new_y = y - top
    new_x = x - left
    new_xy = torch.tensor([new_y, new_x], device=feature_map.device)

    return cropped, new_xy


def random_rotate_preserving_target(feature_map, xy_coords, angle_range=(-15, 15)):
    """
    feature_map: Tensor[H, W, C]
    xy_coords: [x, y]  (column, row)
    """
    # 1) sample angle
    angle = float(torch.empty(1).uniform_(angle_range[0], angle_range[1]))

    # 2) unpack dims & center
    H, W, C = feature_map.shape
    cx, cy = W / 2.0, H / 2.0

    # 3) original x,y
    x, y = xy_coords

    # 4) rotate coordinate
    θ = math.radians(angle)
    cosθ = math.cos(θ)
    sinθ = math.sin(θ)

    dx = x - cx
    dy = y - cy

    new_x = cosθ * dx - sinθ * dy + cx
    new_y = sinθ * dx + cosθ * dy + cy

    # 5) check bounds
    if not (0 <= new_x < W and 0 <= new_y < H):
        # would leave image: skip
        return feature_map, xy_coords

    # 6) perform the actual rotation
    #    TF.rotate expects C×H×W, CCW by 'angle' degrees
    fm_CHW = feature_map.permute(2, 0, 1)
    rotated_CHW = TF.rotate(fm_CHW, angle, expand=False, center=(cx, cy))
    rotated_HWC = rotated_CHW.permute(1, 2, 0)

    return rotated_HWC, [new_x, new_y]


"""
Naming utils
"""
# compile once at module load
PLACEHOLDER_PATTERN = re.compile(r"<person(\d+)>")

def count_unique_people(strings: List[str]) -> int:
    """
    Given a list of strings each containing exactly one "<person{number}>",
    returns the count of distinct person IDs.
    """
    pattern = re.compile(r'<person(\d+)>')
    seen = set()

    for s in strings:
        m = pattern.search(s)
        if m:
            seen.add(int(m.group(1)))

    return len(seen)

def collect_placeholders(value: Any, found: Set[str]) -> None:
    """
    Recursively scan `value` (which may be a string or list) 
    for all occurrences of <personN> and add "personN" to `found`.
    """
    if isinstance(value, str):
        for m in PLACEHOLDER_PATTERN.finditer(value):
            found.add(f"person{m.group(1)}")
    elif isinstance(value, list):
        for v in value:
            collect_placeholders(v, found)
    # other types are ignored

def replace_in_value(
    value: Any,
    placeholder_to_name: Dict[str, str]
) -> Any:
    """
    Recursively replace all occurrences of <personN> in `value`
    with the corresponding name from placeholder_to_name.
    """
    if isinstance(value, str):
        def _repl(m):
            key = f"person{m.group(1)}"
            return placeholder_to_name.get(key, m.group(0))
        return PLACEHOLDER_PATTERN.sub(_repl, value)

    elif isinstance(value, list):
        return [replace_in_value(v, placeholder_to_name) for v in value]

    else:
        return value

def find_sorted_placeholders(episode: dict) -> List[str]:
    """
    Return a numerically-sorted list of unique placeholder keys
    (e.g. ["person1", "person2", ...]) found anywhere in the episode.
    """
    found: Set[str] = set()
    for v in episode.values():
        collect_placeholders(v, found)
    return sorted(found, key=lambda p: int(p.replace("person", "")))
