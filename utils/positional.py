import torch
import torch.nn as nn
import torch.nn.functional as F

class Learnable2DPositionalEncodingMax(nn.Module):
    def __init__(self, E: int = 768, H_max: int = 50, W_max: int = 50):
        """
        2D learnable positional encoding with a “master grid” (H_max x W_max).
        E must be even.
        """
        super().__init__()
        assert E % 2 == 0, "Embedding dimension E must be even"
        self.E = E
        self.E_half = E // 2
        self.H_max = H_max
        self.W_max = W_max

        # Learnable parameters: row_embed (H_max x E/2) and col_embed (W_max x E/2)
        self.row_embed = nn.Parameter(torch.randn(H_max, self.E_half))
        self.col_embed = nn.Parameter(torch.randn(W_max, self.E_half))

    def forward(self, H: int, W: int):
        """
        Returns a tensor of shape (H, W, E) interpolated from (H_max, W_max, E).
        If H <= H_max and W <= W_max, we simply slice and concatenate.
        """
        # 1) Compose a non-interpolated “master grid”: (H_max, W_max, E)
        #    - For each i < H_max, j < W_max: PE[i,j] = [ row_embed[i] ; col_embed[j] ]
        #    - Use broadcasting:
        row_part = self.row_embed.unsqueeze(1)    # (H_max, 1, E/2)
        col_part = self.col_embed.unsqueeze(0)    # (1, W_max, E/2)
        pe_master = torch.cat([
            row_part.expand(-1, self.W_max, -1),  # (H_max, W_max, E/2)
            col_part.expand(self.H_max, -1, -1)   # (H_max, W_max, E/2)
        ], dim=-1)  # Result: (H_max, W_max, E)

        # 2) If H and W are exact (no interpolation)
        if H == self.H_max and W == self.W_max:
            return pe_master

        # 3) Otherwise, interpolate from (H_max, W_max, E) to (H, W, E)
        #    F.interpolate works on 4D tensors [N,C,H,W], where C=channels
        #    => change (H_max,W_max,E) to (1, E, H_max, W_max) for interpolation
        pe_4D = pe_master.permute(2, 0, 1).unsqueeze(0)  # (1, E, H_max, W_max)
        pe_4D_interpolated = F.interpolate(
            pe_4D,
            size=(H, W),
            mode='bicubic',            # or 'bilinear'
            align_corners=False
        )  # (1, E, H, W)
        # Return to (H, W, E)
        pe_hw = pe_4D_interpolated.squeeze(0).permute(1, 2, 0)  # (H, W, E)
        return pe_hw
