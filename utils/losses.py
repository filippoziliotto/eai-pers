
# Library imports
from torch.nn import functional as F
import torch.nn as nn
import torch

# Local imports
from utils.heatmap import generate_gt_heatmap

"""
Loss Utils
"""

def compute_loss(gt_coords, output, loss_choice='L2', feature_map=None):
    """
    Computes the loss between ground truth and predicted coordinates.
    
    Args:
        ground_truth_coords (Tensor): Ground truth coordinates.
        predicted_coords (Tensor): Predicted coordinates.
        loss_choice (str): Either 'L1' for Manhattan loss or 'L2' for Euclidean loss.
        
    Returns:
        loss (Tensor): Computed loss.
    """
    
    if loss_choice not in ['L1', 'L2', 'Huber', 'Huber+Heatmap', "Chebyschev-cobined"]:
        assert 'coords' in output, "'coords' key missing in output for selected loss_choice"
    elif loss_choice not in ['Heatmap', 'SCE']:
        assert 'value_map' in output, "'value_map' key missing in output for selected loss_choice"
    
    # Regression
    if loss_choice == 'L1':
        return L1_loss(output["coords"], gt_coords)
    elif loss_choice == 'L2':
        return L2_loss(output["coords"], gt_coords)
    elif loss_choice == 'Huber':
        return Huber_loss(output["coords"], gt_coords)
    
    # Classification
    elif loss_choice == 'CE':
        gt_heatmap = generate_gt_heatmap(gt_coords, feature_map)
        pred_heatmap = output["value_map"]
        return CE_loss(pred_heatmap, gt_heatmap)
    elif loss_choice == 'Huber+CE':
        gt_heatmap = generate_gt_heatmap(gt_coords, feature_map, True)
        pred_heatmap = output["value_map"]
        return Huber_loss(output["coords"], gt_coords) + CE_loss(pred_heatmap, gt_heatmap)
    elif loss_choice == 'SCE':
        # Scaled Cross-Entropy loss
        gt_heatmap = generate_gt_heatmap(gt_coords, output["value_map"])
        pred_heatmap = output["value_map"]
        return ScaledCE_loss(pred_heatmap, gt_heatmap, output["dist_matrix"])
    
    # Combined Chebyshev loss
    # loss = alpha * L_in + beta * L_far + gamma * L_near
    elif loss_choice == "Chebyshev-combined":
        alpha, beta, gamma = 1.0, 1.0, 1.0
        return Chebyshev_combined_loss(
            pred_coords=output["coords"],
            gt_coords=gt_coords,
            F_input=output["value_map"],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            eps=1e-6
        )
    
    else:
        raise ValueError(f"Unknown loss choice: {loss_choice}. Use 'L1' or 'L2', 'Huber'.")
    
    
def L1_loss(pred_coords, gt_coords):
    """
    Computes Mean Absolute Error (L1) loss between the predicted coordinates 
    from the global regression branch and the ground truth coordinates.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (b, 2).
        gt_coords (torch.Tensor): Ground truth coordinates of shape (b, 2).
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    return F.l1_loss(pred_coords, gt_coords)

def L2_loss(pred_coords, gt_coords):
    """
    Computes Mean Squared Error (L2) loss between the predicted coordinates 
    from the global regression branch and the ground truth coordinates.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (b, 2).
        gt_coords (torch.Tensor): Ground truth coordinates of shape (b, 2).
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    return F.mse_loss(pred_coords, gt_coords)

def Huber_loss(pred_coords, gt_coords, delta=1.0):
    """
    Computes Huber loss between the predicted coordinates and ground truth coordinates.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (b, 2).
        gt_coords (torch.Tensor): Ground truth coordinates of shape (b, 2).
        delta (float): The point where the loss function changes
            from quadratic to linear. Default is 1.0.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    return nn.SmoothL1Loss(beta=delta)(pred_coords, gt_coords)

def CE_loss(
    pred_heatmap: torch.Tensor,
    gt_heatmap:   torch.Tensor,
    reduction:    str = 'mean'
) -> torch.Tensor:
    """
    “Normal” (categorical) cross‐entropy over the entire H×W grid,
    supporting soft‐labels {0.0, 0.5, 1.0} on “inside” pixels and ignoring 
    all “outside” pixels marked as -1 in gt_heatmap.

    Args:
        pred_heatmap (Tensor):
            Predicted raw logits, shape (b, H, W, 1).
        gt_heatmap (Tensor):
            GT soft‐labels, shape (b, H, W, 1). Valid inside‐map values 
            are {0.0, 0.5, 1.0}, and outside‐map pixels must be exactly -1.0.
        reduction (str):
            One of {'mean', 'sum', 'none'}. 
            - 'mean': returns scalar = (Σ_{k} L^{(k)}) / b 
                      (where L^{(k)} is the per‐sample loss).  
            - 'sum' : returns scalar = Σ_{k} L^{(k)}.  
            - 'none' : returns a tensor of shape (b,) with each sample’s loss.

    Returns:
        Tensor: 
          If reduction in {'mean','sum'}, returns a scalar tensor. 
          If reduction=='none', returns a tensor of shape (b,) containing each sample’s loss.
    """
    b, H, W, _ = pred_heatmap.shape
    device = pred_heatmap.device

    # 1) Flatten both pred_heatmap and gt_heatmap to shape (b, H*W)
    pred_flat = pred_heatmap.view(b, H * W)       # raw logits, shape (b, H*W)
    gt_flat   = gt_heatmap.view(b, H * W)         # targets ∈ {1, 0.5, 0, -1}

    # 2) Build a mask of “inside” pixels: gt_flat >= 0.0
    #    (outside pixels were marked -1.0, so we ignore those)
    mask_inside = (gt_flat >= 0.0)  # shape (b, H*W), dtype=torch.bool

    # 3) For each sample k, compute the sum of all positive GT‐values at inside pixels
    #    This is the normalizing constant so that the remaining GTs become a valid distribution.
    #    E.g. center=1, four neighbors=0.5 each → sum = 1 + 4*0.5 = 3.0  
    #    If a sample has no “inside” pixels at all (mask_inside.sum()==0), we can set loss=0 for that sample.
    gt_flat_positive = torch.clamp(gt_flat, min=0.0)  # anything negative becomes 0
    # shape (b,) = sum over the H*W dimension
    sum_positive_per_sample = gt_flat_positive.sum(dim=1)  # may be 0 if everything is “outside.”

    # 4) Build the normalized target distribution ˜G^{(k)}:
    #    ˜G^{(k)}_{i} = gt_flat_positive[k,i] / sum_positive_per_sample[k]
    #                 if gt_flat_positive[k,i] > 0, else 0.
    #    If sum_positive_per_sample[k]==0, we will just skip that sample.
    #    We create a (b, H*W) tensor `gt_normalized` that holds ˜G.
    eps = 1e-8  # to avoid division by zero
    normalizer = sum_positive_per_sample.unsqueeze(1).clamp(min=eps)  # shape (b,1)
    gt_normalized = gt_flat_positive / normalizer  # shape (b, H*W)

    # 5) Compute log‐softmax of pred_flat along dimension=1 (over the H*W “classes”)
    log_probs = F.log_softmax(pred_flat, dim=1)  # shape (b, H*W)

    # 6) The per‐sample cross‐entropy is
    #     L^{(k)} = - Σ_{i=1}^{H*W} ˜G^{(k)}_{i} * log_probs[k, i],
    #    but we only want to sum over “inside” pixels (mask_inside[k, i]==True).  
    #    Since ˜G_{i} is already zero on “outside” pixels, we can just do:
    #        L^{(k)} = - Σ_{i} [ gt_normalized[k,i] * log_probs[k,i] ] 
    #    and those terms are zero wherever gt_normalized==0.
    per_pixel_term = gt_normalized * log_probs  # shape (b, H*W)
    loss_per_sample = - per_pixel_term.sum(dim=1)  # shape (b,)

    # 7) For any sample k where sum_positive_per_sample[k] was zero (no inside pixels),
    #    we define loss_per_sample[k] = 0.0 explicitly.
    no_inside_mask = (sum_positive_per_sample < eps)  # shape (b,)
    if no_inside_mask.any():
        loss_per_sample = loss_per_sample.masked_fill(no_inside_mask, 0.0)

    # 8) Finally, apply reduction across the batch
    if reduction == 'mean':
        # mean over the b samples
        return loss_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum()

def ScaledCE_loss(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    dist_matrix: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes a scaled cross-entropy loss between predicted and ground truth heatmaps.

    Args:
        pred_heatmap (torch.Tensor): Predicted heatmap tensor of shape (N, H, W, 1) or (N, H, W).
        gt_heatmap (torch.Tensor): Ground truth heatmap tensor of shape (N, H, W, 1) or (N, H, W).
        dist_matrix (torch.Tensor): Distance matrix tensor of shape (N, H, W, 1) or (N, H, W).
        reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum'.

    Returns:
        torch.Tensor: The computed scaled cross-entropy loss (scalar).
    """

    # Squeeze to (n, H, W) and Flatten the matrices
    pred_heatmap_flat = pred_heatmap.squeeze(-1).view(pred_heatmap.size(0), -1)
    gt_heatmap_flat = gt_heatmap.squeeze(-1).view(gt_heatmap.size(0), -1)
    dist_matrix_flat = dist_matrix.squeeze(-1).view(dist_matrix.size(0), -1)
    
    # Cross-Entropy loss manually
    # ce_loss = gt_heatmap_flat * F.log_softmax(pred_heatmap_flat, dim=1)
    
    # Binary cross entropy with reduction none
    ce_loss = F.binary_cross_entropy_with_logits(pred_heatmap_flat, gt_heatmap_flat, reduction='none')
    
    # Max between dist_matrix and gt_heatmap
    max_dist = torch.max(dist_matrix_flat, gt_heatmap_flat)
    
    # Scale the CE loss by the max distance
    scaled_ce_loss = torch.sum(ce_loss * max_dist, dim=-1)
    
    if reduction == 'mean':
        return scaled_ce_loss.mean()
    elif reduction == 'sum':
        return scaled_ce_loss.sum()
    else:
        raise ValueError(f"Unknown reduction method: {reduction}. Use 'mean' or 'sum'.")
    
def Chebyshev_combined_loss(
    pred_coords,   # LongTensor, shape (b,2), each row=(y_pred, x_pred)
    gt_coords,     # LongTensor, shape (b,2), each row=(y_gt, x_gt)
    F_input,       # FloatTensor, shape (b, H, W, E), zero outside
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    eps=1e-6
):
    """
    Returns:
      L_total = alpha*L_in + beta*L_far + gamma*L_near

    pred_coords: (b,2)  = predicted (y,x) per batch
    gt_coords:   (b,2)  = ground‐truth (y,x) per batch
    F_input:   (b,H,W,E) = feature‐map whose zero‐vectors mark outside

    L_in   = average_n [-log( M_in[y_pred, x_pred] + eps )]
    L_far  = average_n [ max(0, d_cheb - 2)^2 ]
    L_near = average_n [ 0 if d_cheb <=1, else (d_cheb - 1)^2 ]

    where
      d_cheb = max(|y_pred - y_gt|, |x_pred - x_gt|).
    """
    b, H, W, E = F_input.shape
    device = F_input.device

    # 1) Build binary “inside‐map” mask M_in[n,i,j] ∈ {0,1}
    F_norm = torch.norm(F_input, dim=-1)    # (b, H, W)
    M_in   = (F_norm > 0).float()            # (b, H, W)

    # 2) Index M_in at the predicted coordinates → m_n = 0 or 1
    ys_pred = pred_coords[:, 0].clamp(0, H-1)
    xs_pred = pred_coords[:, 1].clamp(0, W-1)
    batch_idx = torch.arange(b, device=device)

    m = M_in[batch_idx, ys_pred.round().int(), xs_pred.round().int()]    # shape (b,)

    # Inside‐map loss:  L_in = -(1/b) sum_n log(m_n + eps)
    L_in = - (m + eps).clamp(min=eps).log().mean()

    # 3) Compute Chebyshev distance d_cheb[n] between pred and GT
    ys_gt = gt_coords[:, 0].clamp(0, H-1)
    xs_gt = gt_coords[:, 1].clamp(0, W-1)
    d_cheb = torch.max(
        torch.abs(ys_pred - ys_gt),
        torch.abs(xs_pred - xs_gt)
    )  # shape (b,)

    # 4) “Far‐inside” loss:  L_far = average_n [ max(0, d_cheb - 2)^2 ]
    far_term = torch.relu(d_cheb - 2.0) ** 2
    L_far = far_term.mean()

    # 5) “Near‐ring” loss:  L_near = average_n [ 0 if d_cheb ≤1 else (d_cheb - 1)^2 ]
    near_term = torch.where(
        d_cheb <= 1.0,
        torch.zeros_like(d_cheb),
        (d_cheb - 1.0) ** 2
    )  # shape (b,)
    L_near = near_term.mean()

    # 6) Combine
    L_total = alpha * L_in + beta * L_far + gamma * L_near
    return L_total