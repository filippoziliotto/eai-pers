# Qua va messo un attimo di preprocessing
# evaneutalmente anche roba di collate_fn o rba cosi
from typing import List, Tuple, Dict, Any, Union
from dataset.maps.base_map import BaseMap


def map_to_xyz(episode: Dict, map: BaseMap) -> List[float]:
    """
    Converts the target coordinates in the 2D map frame coordinates (x, y) 
    to (x, y, z) in the global frame in the habitat view.
    
    Return [x,y,z] in the global frame.
    """
    
    # episode["object_pos"], episode["robot_xyz"], episode["robot_xy"], episode["robot_heading"]
    
    
    return [0, 0, 0]  # Placeholder
    
def xyz_to_map(xyz_target, robot_xyz, robot_xy, robot_heading):
    """
    Converts the target coordinates in the global frame (x, y, z) to the 2D map frame coordinates (x, y).
    
    Return [x,y] in the map frame.
    """
    return [0, 0]  # Placeholder

    