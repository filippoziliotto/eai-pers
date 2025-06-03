import torch
import torch.nn as nn

class LearnableScalarSimilarity(nn.Module):
    """
    Unbounded learned similarity: projects each E-dimensional vector
    to a single scalar (no normalization), then multiplies.
    
    f_maps: Tensor of shape (b, h, w, E)
    q_emb:  Tensor of shape (b, E)
    
    Returns sim_map of shape (b, h, w), where
        sim_map[b,i,j] = feature_proj(f_maps[b,i,j,:]) * query_proj(q_emb[b,:])
    """
    def __init__(self, input_dim):
        """
        Args:
            input_dim: Dimensionality E of both feature and query embeddings.
        """
        super().__init__()
        # Both projections map from E → 1 (no bias).
        self.feature_proj = nn.Linear(input_dim, 1, bias=False)
        self.query_proj   = nn.Linear(input_dim, 1, bias=False)

    def forward(self, f_maps: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_maps: Tensor of shape (b, h, w, E)
            q_emb:  Tensor of shape (b, E)
        
        Returns:
            sim_map: Tensor of shape (b, h, w)
        """
        b, h, w, E = f_maps.shape
        
        # 1) Project each spatial feature to one scalar:
        #    Flatten spatial dims so Linear can be applied in batch.
        f_flat = f_maps.view(b * h * w, E)       # → (b*h*w, E)
        f_proj = self.feature_proj(f_flat)       # → (b*h*w, 1)
        f_proj = f_proj.view(b, h, w)            # → (b, h, w)

        # 2) Project query to one scalar per batch element:
        q_proj = self.query_proj(q_emb)          # → (b, 1)
        q_proj = q_proj.view(b)                  # → (b,)

        # 3) Expand query scalar to (b, h, w) and multiply:
        #    Each location’s score = f_proj[b,i,j] * q_proj[b]
        q_grid = q_proj.view(b, 1, 1).expand(-1, h, w)  # → (b, h, w)
        sim_map = f_proj * q_grid                        # → (b, h, w)

        return sim_map
