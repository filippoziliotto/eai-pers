
# Internal imports
import os
import cv2
import numpy as np

def visualize(
    query: str, 
    gt_target: np.ndarray, 
    value_map,  # can be a torch.Tensor or np.ndarray
    map_path: str,
    batch_idx: int,
    name: str = "prediction",
    split: str = "train",
    use_obstacle_map: bool = False,
    upscale_factor: float = 1.0
) -> None:
    """
    Creates a visualization by combining the value map (with annotations) and, if requested,
    the obstacle map. Circles indicating the ground truth (GT) and the maximum value (Max)
    are drawn on both images. A top margin is added for the query text, and the final image
    is saved using a naming scheme.
    """
    # Convert the value map to numpy if needed.
    if hasattr(value_map, "requires_grad") and value_map.requires_grad:
        value_map = value_map.detach().cpu().numpy()
    elif hasattr(value_map, "cpu"):
        value_map = value_map.cpu().numpy()
    
    # Ensure ground truth is a numpy int32 array.
    if hasattr(gt_target, "cpu"):
        gt_target_ = gt_target.cpu().numpy().astype(np.int32)
    else:
        gt_target_ = gt_target.astype(np.int32)
    
    # Compute the pixel with the maximum value.
    max_pixel = find_index_max_value(value_map)
    
    # Convert the value map into an RGB image using the Inferno colormap.
    value_map_img = monochannel_to_inferno_rgb(value_map)
    
    # add circles on the value map.
    value_map_img = add_circles_on_image(value_map_img, gt_target, max_pixel)
    combined_image = value_map_img

    # ----- Add top margin for query text -----
    final_image = add_top_margin(combined_image, margin_height=15, query_text=query, font_scale=0.3, thickness=1)
    
    # ----- Upscale the final image if requested -----
    if upscale_factor != 1.0:
        final_image = cv2.resize(final_image, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
    
    # ----- Save the final image -----
    save_image_to_disk(final_image, base_path=f"trainer/visualizations/{split}", name=name, idx_=batch_idx)


def find_index_max_value(value_map: np.ndarray) -> np.ndarray:
    """
    Find the pixel location with the maximum value in the value map.
    """
    max_index = np.unravel_index(np.argmax(value_map, axis=None), value_map.shape)
    # Adjust ordering if needed.
    return np.flipud(max_index)

def add_circles_on_image(image: np.ndarray, gt_target, max_pixel) -> np.ndarray:
    """
    Draws a red circle (with label "GT") for the ground truth pixel and
    a yellow circle (with label "Max") for the maximum value pixel.
    
    Args:
        image: The input image.
        gt_target: The (x, y) coordinates for the ground truth pixel.
        max_pixel: The (x, y) coordinates for the maximum value pixel.
    
    Returns:
        The image with the circles and labels added.
    """
    # Draw a red circle and label for GT.
    image = cv2.circle(image, (int(gt_target[0]), int(gt_target[1])), 3, (0, 0, 255), 2)
    image = cv2.putText(image, "GT", (int(gt_target[0]) + 5, int(gt_target[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Draw a yellow circle and label for Max.
    image = cv2.circle(image, (int(max_pixel[0]), int(max_pixel[1])), 3, (0, 255, 255), 2)
    image = cv2.putText(image, "Max", (int(max_pixel[0]) + 5, int(max_pixel[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    
    return image


def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """
    Normalize a single-channel image and apply the Inferno colormap.
    """
    min_val, max_val = np.min(image), np.max(image)
    if max_val - min_val == 0:
        normalized = np.zeros_like(image, dtype=np.uint8)
    else:
        normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return colored


def crop_map_borders(image: np.ndarray) -> tuple:
    """
    Compute the crop boundaries based on nonzero pixels in the image.
    
    Returns:
        A tuple (x_min, x_max, y_min, y_max) representing the bounding box.
        If no nonzero pixels are found, returns full image boundaries.
    """
    # Convert to grayscale if the image is colored.
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    coords = cv2.findNonZero(gray)
    if coords is None:
        return (0, image.shape[1], 0, image.shape[0])
    
    x, y, w, h = cv2.boundingRect(coords)
    return (x, x + w, y, y + h)


def crop_image(image: np.ndarray, crop_coords: tuple) -> np.ndarray:
    """
    Crop an image using the specified coordinates.
    
    Args:
        image: The image to crop.
        crop_coords: A tuple (x_min, x_max, y_min, y_max).
    
    Returns:
        The cropped image.
    """
    x_min, x_max, y_min, y_max = crop_coords
    return image[y_min:y_max, x_min:x_max]


def add_top_margin(
    image: np.ndarray, 
    margin_height: int, 
    query_text: str, 
    font_scale: float = 0.5, 
    thickness: int = 1
) -> np.ndarray:
    """
    Adds a top margin to the image and writes the query text in that space.
    
    Args:
        image: The input image.
        margin_height: The height (in pixels) of the margin.
        query_text: The text to add in the margin.
        font_scale: Scale factor for the text.
        thickness: Thickness of the text strokes.
    
    Returns:
        The image with the added top margin.
    """
    height, width = image.shape[:2]
    margin = np.zeros((margin_height, width, 3), dtype=np.uint8)
    cv2.putText(margin, f"Query: {query_text}", (5, margin_height - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    final_image = np.concatenate((margin, image), axis=0)
    return final_image


def save_image_to_disk(image: np.ndarray, base_path: str = "trainer/visualizations/", name: str = "prediction", idx_: int = 0) -> None:
    """
    Saves the image to disk with the naming scheme: <name>_<idx_>.png.
    """
    os.makedirs(base_path, exist_ok=True)
    filepath = os.path.join(base_path, f"{name}_{idx_}.png")
    cv2.imwrite(filepath, image)


"""
Quick visualization
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_value_map(value_map, pred_target, gt_target, filename="trainer/visualizations/value_map.png"):
    """
    Plots the value map with ground truth and predicted target positions.
    Args:
        value_map: Tensor or numpy array of shape (N, H, W, 1) or (N, H, W)
        pred_target: Tensor or numpy array of shape (N, 2)
        gt_target: Tensor or numpy array of shape (N, 2)
        filename: Output filename for the plot
    """
    x = 2
    if hasattr(value_map, 'cpu'):
        map_img = value_map[x].squeeze(-1).cpu().detach().numpy()
    else:
        map_img = value_map[x].squeeze(-1)
    if hasattr(pred_target, 'cpu'):
        pred = pred_target[x].cpu().detach().numpy()
    else:
        pred = pred_target[x]
    if hasattr(gt_target, 'cpu'):
        gt = gt_target[x].cpu().detach().numpy()
    else:
        gt = gt_target[x]

    fig, ax = plt.subplots()
    ax.imshow(map_img, cmap='gray')
    # Draw a red circle at the ground truth position
    circle_gt = patches.Circle((gt[1], gt[0]), radius=1, edgecolor='red', facecolor='none', linewidth=2, label='GT')
    ax.add_patch(circle_gt)
    # Draw a blue cross at the predicted position
    ax.plot(pred[1], pred[0], 'bx', markersize=10, label='Pred')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
