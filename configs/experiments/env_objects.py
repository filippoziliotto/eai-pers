# -------------------------------------------------------------------
# env_objects.py
# Rooms and object masks for embodied AI experiments
# -------------------------------------------------------------------

rooms = ["kitchen", "living_room", "bedroom", "bathroom", "hall"]

object_mask = {
    "pillow":       [0, 0, 1, 0, 0],  # bedroom
    "fridge":       [1, 0, 0, 0, 0],  # kitchen
    "sofa":         [0, 1, 0, 0, 0],  # living room
    "lamp":         [0, 0.5, 0.5, 0, 0],  # bedroom or living room
    "toilet":       [0, 0, 0, 1, 0],  # bathroom
    "bed":          [0, 0, 1, 0, 0],  # bedroom
    "dining_table": [0.8, 0.2, 0, 0, 0],  # kitchen or living room
    "towel":        [0, 0, 0, 1, 0],  # bathroom
    "mirror":       [0, 0, 0.5, 0.5, 0],  # bedroom or bathroom
    "shoe_rack":    [0, 0, 0, 0, 1],  # hall
}

# eventualmente puoi aggiungere funzioni helper
def get_object_mask(obj_name: str):
    return object_mask.get(obj_name, [0]*len(rooms))
