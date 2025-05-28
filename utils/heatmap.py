import torch
import torch.nn.functional as F


def generate_gt_heatmap(gt_coords, value_map):
    """
    Generate a heatmap with 1.0 at gt_coords and 0.0 elsewhere.

    Args:
        gt_coords (torch.Tensor): Tensor of shape (b, 2) with (y, x) coordinates for each batch.
        value_map (torch.Tensor): Tensor of shape (b, H, W, 1) to provide spatial dimensions.

    Returns:
        torch.Tensor: Heatmap of shape (b, H, W, 1) with 1.0 at gt_coords and 0.0 elsewhere.
    """
    b, H, W, _ = value_map.shape
    device = value_map.device

    # start with zeros
    heatmaps = torch.zeros((b, 1, H, W), device=device)

    # mark centers
    ys = gt_coords[:, 0].long().clamp(0, H-1)
    xs = gt_coords[:, 1].long().clamp(0, W-1)
    batch_idx = torch.arange(b, device=device)
    heatmaps[batch_idx, 0, ys, xs] = 1.0

    return heatmaps.permute(0, 2, 3, 1)  # (b, H, W, 1)