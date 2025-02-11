
# Internal imports
import cv2
import numpy as np
import torch
import os

# External imports
from dataset.utils import load_obstacle_map

def find_index_max_value(value_map: np.ndarray) -> np.ndarray:
    """Find the maximum value in a 3D array.

    Args:
        value_map (numpy.ndarray): The input 3D array.

    Returns:
        numpy.ndarray: The index of the maximum value in the 3D array.
    """
    max_val = np.unravel_index(np.argmax(value_map, axis=None), value_map.shape)
    return np.flipud(max_val)

def add_circles_on_image(image: np.ndarray, gt_target: np.ndarray, max_val: np.array) -> np.ndarray:
    """Add a blue circle on the predicted pixel and a red circle on the ground truth pixel.
    Then we also add a yellow circle on the pixel with the maximum value in the value map.

    Args:
        image (numpy.ndarray): The input image.
        gt_target (numpy.ndarray): The ground truth pixel coordinates.
        max_val (numpy.ndarray): The pixel coordinates with the maximum value in the value map.

    Returns:
        numpy.ndarray: The image with the circles added.
    """
    
    # Add a red circle on the ground truth pixel
    image_rgb = cv2.circle(image, (gt_target[0], gt_target[1]), 5, (0, 0, 255), -1)
    image_rgb = cv2.putText(image_rgb, "GT", (gt_target[0]+5, gt_target[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Add a yellow circle on the pixel with the maximum value in the value map
    image_rgb = cv2.circle(image_rgb, (max_val[0], max_val[1]), 5, (0, 255, 255), -1)
    image_rgb = cv2.putText(image_rgb, "Max", (max_val[0]+5, max_val[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

    return image_rgb

def add_query_to_image(image: np.ndarray, text: str) -> np.ndarray:
    """Add text to the upper left corner of an image.

    Args:
        image (numpy.ndarray): The input image.
        text (str): The text to add to the image.

    Returns:
        numpy.ndarray: The image with the text added to the upper left corner.
    """

    # Add the text to the image
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image

def crop_map_borders(value_map: np.ndarray) -> np.ndarray:
    # TODO:
    pass

def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a monochannel float32 image to an RGB representation using the Inferno
    colormap.

    Args:
        image (numpy.ndarray): The input monochannel float32 image.

    Returns:
        numpy.ndarray: The RGB image with Inferno colormap.
    """
    # Normalize the input image to the range [0, 1]
    min_val, max_val = np.min(image), np.max(image)
    peak_to_peak = max_val - min_val
    if peak_to_peak == 0:
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / peak_to_peak

    # Apply the Inferno colormap
    inferno_colormap = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    return inferno_colormap

def save_image_to_disk(image: np.ndarray, base_path: str="trainer/visualizations/", name: str="image", idx_:int=0) -> None:
    """Save an image to disk.
    Args:
        image (numpy.ndarray): The image to save.
        base_path (str, optional): The base path to save the image to. Defaults to "trainer/visualizations/".
        name (str, optional): The name of the image file. Defaults to "image.png".
    """
    cv2.imwrite(f"{base_path}{name}_{idx_}.png", image)
    
    
query = "find my mac"
gt_target = torch.tensor([100,100])
pred_target = torch.tensor([50,50])
value_map  = torch.rand(100, 100)

# Move imahe to numpy
value_map = value_map.cpu().numpy()
gt_target = gt_target.cpu().numpy()


def visualize(
    query: str, 
    gt_target: np.ndarray, 
    value_map: np.ndarray,
    map_path: str,
    batch_idx: int,
    name: str="map",
    overlay_obstacle_map: bool=False
    ) -> None:
    """Visualize the image, query, ground truth and predicted pixel, and the value map.

    Args:
        image (numpy.ndarray): The input image.
        query (str): The query text.
        gt_target (numpy.ndarray): The ground truth pixel coordinates.
        pred_target (np.ndarray): The predicted pixel coordinates.
        value_map (np.ndarray): The value map.
    """
    
    value_map = value_map.detach().cpu().numpy() if value_map.requires_grad else value_map.cpu().numpy()
    gt_target = gt_target.cpu().numpy().astype(np.int32)
    
    # Find the pixel with the maximum value in the value map
    max_val = find_index_max_value(value_map)
    
    # Convert the value map to an RGB image 
    image_rgb = monochannel_to_inferno_rgb(value_map)
    
    # Add circles on the image
    image_rgb = add_circles_on_image(image_rgb, gt_target, max_val)
    
    # Add the query to the image
    image_rgb = add_query_to_image(image_rgb, query)
    
    if overlay_obstacle_map:
        # Load the obstacle map and process it
        obstacle_map_filepath = os.path.join(map_path, "obstacle_map.npy")
        obstacle_map = load_obstacle_map(obstacle_map_filepath)

        # Convert obstacle map to 8-bit and then to an RGB representation
        obstacle_map_8u = cv2.convertScaleAbs(obstacle_map)
        obstacle_map_rgb = cv2.cvtColor(obstacle_map_8u, cv2.COLOR_GRAY2RGB)        
    
        # overlay obstacle map on value map
        image_rgb = image_rgb + obstacle_map_rgb
    
    # Save the images to disk
    save_image_to_disk(image_rgb, name=name, idx_=batch_idx)
    