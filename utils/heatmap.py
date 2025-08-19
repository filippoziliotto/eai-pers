import torch
import torch.nn.functional as F


def generate_gt_heatmap(gt_coords, feature_map, use_neighbors=True):
    """
    Generate a ground-truth heatmap of shape (b, H, W, 1) where:
      - All “outside‐map” locations (i.e., zero embeddings in `feature_map`) are set to -1.
      - All valid “inside‐map” locations are initialized to 0.
      - The GT point (y, x) is set to 1.
      - Its 8‐connected neighbors (if use_neighbors=True) are set to 0.5, but only if they are inside the map.

    Args:
        gt_coords (torch.Tensor): Tensor of shape (b, 2), each row is (y, x) for the GT point.
        feature_map (torch.Tensor): Tensor of shape (b, H, W, E). All-zero embeddings mark “outside the map.”
        use_neighbors (bool): If True, assign 0.5 to the 8‐neighbors of each GT point (if inside).

    Returns:
        torch.Tensor: gt_heatmap of shape (b, H, W, 1), dtype=torch.float32.
    """
    b, H, W, _ = feature_map.shape
    device = feature_map.device

    # 1) Compute a mask for “inside‐map” vs “outside‐map” based on feature_map embeddings.
    #    If all E channels at (b, i, j) are zero, we treat it as “outside.”
    #    Otherwise, it’s “inside.”
    #    mask_inside: (b, H, W), True where inside, False where outside.
    with torch.no_grad():
        mask_inside = (feature_map.abs().sum(dim=-1) > 0)  # shape (b, H, W)

    # 2) Initialize a heatmap of shape (b, H, W) with:
    #    -1.0 at all “outside” locations
    #     0.0 at all “inside” locations
    heatmaps = torch.full((b, H, W), -1.0, device=device, dtype=torch.float)
    heatmaps[mask_inside] = 0.0

    # 3) Clamp GT coordinates to ensure they lie within [0, H-1] x [0, W-1].
    ys = gt_coords[:, 0].long().clamp(0, H - 1)  # (b,)
    xs = gt_coords[:, 1].long().clamp(0, W - 1)  # (b,)
    batch_idx = torch.arange(b, device=device)

    # 4) Assign “1.0” to the GT point—but only if that point is marked “inside.”
    valid_center = mask_inside[batch_idx, ys, xs]  # (b,) boolean
    if valid_center.any():
        yc = ys[valid_center]
        xc = xs[valid_center]
        bc = batch_idx[valid_center]
        heatmaps[bc, yc, xc] = 1.0

    if use_neighbors:
        # 5) Define 8‐connected neighbor offsets and their label value (0.5).
        offset_value_pairs = [
            ((-1, -1), 0.5), ((-1, 0), 0.5), ((-1, 1), 0.5),
            ((0, -1), 0.5),                ((0, 1), 0.5),
            ((1, -1), 0.5),  ((1, 0), 0.5),  ((1, 1), 0.5),
        ]

        # 6) For each offset, compute neighbor indices (clamped), then only assign 0.5 if that neighbor is “inside.”
        for (dy, dx), value in offset_value_pairs:
            yn = (ys + dy).clamp(0, H - 1)  # (b,)
            xn = (xs + dx).clamp(0, W - 1)  # (b,)
            # Check which of these (b, yn, xn) are truly inside the map:
            valid_neighbor = mask_inside[batch_idx, yn, xn]  # (b,) boolean
            if not valid_neighbor.any():
                continue

            bn = batch_idx[valid_neighbor]
            yv = yn[valid_neighbor]
            xv = xn[valid_neighbor]

            # Only overwrite if the new value (0.5) is greater than the current heatmap value.
            current_vals = heatmaps[bn, yv, xv]
            new_vals = torch.full_like(current_vals, value, dtype=torch.float)
            heatmaps[bn, yv, xv] = torch.maximum(current_vals, new_vals)

    # 7) Reshape to (b, H, W, 1) before returning
    return heatmaps.unsqueeze(-1)  # shape (b, H, W, 1)
