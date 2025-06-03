import torch
import torch.nn.functional as F


def generate_gt_heatmap(gt_coords, value_map, feature_map, use_neighbors=False):
    """
    Generate a heatmap with a discrete Gaussian-like pattern:
    - 1.0 at center
    - 0.5 at 8-connected neighbors
    - 0.2 at 16-connected second ring

    Args:
        gt_coords (torch.Tensor): Tensor of shape (b, 2) with (y, x) coordinates.
        value_map (torch.Tensor): Tensor of shape (b, H, W, 1) for spatial dimensions.
        use_neighbors (bool): If True, sets values in a 5x5 pattern centered at gt_coords.

    Returns:
        torch.Tensor: Heatmap of shape (b, H, W, 1).
    """
    b, H, W, _ = value_map.shape
    device = value_map.device

    heatmaps = torch.zeros((b, 1, H, W), device=device)

    ys = gt_coords[:, 0].long().clamp(0, H - 1)
    xs = gt_coords[:, 1].long().clamp(0, W - 1)
    batch_idx = torch.arange(b, device=device)

    # Set center value
    heatmaps[batch_idx, 0, ys, xs] = 1.0

    if use_neighbors:
        # Define offsets and corresponding values
        offset_value_pairs = [
            # First ring (8-neighbors)
            ((-1, -1), 0.5), ((-1, 0), 0.5), ((-1, 1), 0.5),
            ((0, -1), 0.5),               ((0, 1), 0.5),
            ((1, -1), 0.5), ((1, 0), 0.5), ((1, 1), 0.5),

            # Second ring (next 16 neighbors at distance 2)
            ((-2, -2), 0.2), ((-2, -1), 0.2), ((-2, 0), 0.2), ((-2, 1), 0.2), ((-2, 2), 0.2),
            ((-1, -2), 0.2),                             ((-1, 2), 0.2),
            ((0, -2), 0.2),                              ((0, 2), 0.2),
            ((1, -2), 0.2),                              ((1, 2), 0.2),
            ((2, -2), 0.2), ((2, -1), 0.2), ((2, 0), 0.2), ((2, 1), 0.2), ((2, 2), 0.2),
        ]

        for (dy, dx), value in offset_value_pairs:
            yn = (ys + dy).clamp(0, H - 1)
            xn = (xs + dx).clamp(0, W - 1)
            heatmaps[batch_idx, 0, yn, xn] = torch.maximum(
                heatmaps[batch_idx, 0, yn, xn],
                torch.full_like(yn, value, dtype=torch.float)
            )

    return heatmaps.permute(0, 2, 3, 1)  # (b, H, W, 1)
