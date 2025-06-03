import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableScalarSimilarity(nn.Module):
    """
    Unbounded learned score per location, implemented as:
      feature_branch: E → (E//2) → (E//4) → 1
      query_branch:   E → (E//2) → (E//4) → 1
    with ReLU activations in between.
    
    f_maps: Tensor of shape (b, h, w, E)
    q_emb:  Tensor of shape (b, E)

    Returns sim_map of shape (b, h, w), where
      sim_map[b,i,j] = feature_mlp(f_maps[b,i,j,:]) * query_mlp(q_emb[b,:])
    """
    def __init__(self, input_dim):
        """
        Args:
            input_dim:  Dimensionality E of both feature and query embeddings.
                        Must be divisible by 4 for the chosen reductions.
        """
        super().__init__()
        hidden1 = input_dim // 2
        hidden2 = input_dim // 4

        # Feature branch (no bias on final layer to match original design, but bias=True on intermediates is OK)
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, 1, bias=False)
        )

        # Query branch
        self.query_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, 1, bias=False)
        )

    def forward(self, f_maps: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_maps: Tensor of shape (b, h, w, E)
            q_emb:  Tensor of shape (b, E)

        Returns:
            sim_map: Tensor of shape (b, h, w)
        """
        b, h, w, E = f_maps.shape

        # 1) Project each spatial feature vector through the MLP → scalar
        f_flat = f_maps.view(b * h * w, E)         # → (b*h*w, E)
        f_proj = self.feature_mlp(f_flat)          # → (b*h*w, 1)
        f_proj = f_proj.view(b, h, w)              # → (b, h, w)

        # 2) Project query through the same‐structure MLP → scalar
        q_proj = self.query_mlp(q_emb)             # → (b, 1)
        q_proj = q_proj.view(b)                    # → (b,)

        # 3) Expand query scalar to (b, h, w) and multiply
        q_grid = q_proj.view(b, 1, 1).expand(-1, h, w)  # → (b, h, w)
        sim_map = f_proj * q_grid                        # → (b, h, w)

        return sim_map
