import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Simplified multi-head attention class con residuo.

        Args:
            embed_dim (int): The embedding dimension (E).
            num_heads (int): The number of attention heads.
            dropout (float): Dropout rate da applicare sull'output di attenzione.
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear projections per Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Proiezione finale
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout e LayerNorm per il residuo
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, Q_input, K_input, V_input):
        """
        Multi-head attention forward pass con residuo.

        Args:
            Q_input: Tensor di forma (batch_size, n_q, embed_dim) – Query input.
            K_input: Tensor di forma (batch_size, k, embed_dim) – Key input.
            V_input: Tensor di forma (batch_size, k, embed_dim) – Value input.

        Returns:
            output: Tensor di forma (batch_size, n_q, embed_dim)
        """
        b, n_q, embed_dim = Q_input.size()
        b2, k, _ = K_input.size()
        assert embed_dim == self.embed_dim and b == b2, "Dimension mismatch tra Q/K e parametri del modello"

        # 1) Proiezioni lineari
        Q = self.q_proj(Q_input)  # (b, n_q, E)
        K = self.k_proj(K_input)  # (b, k, E)
        V = self.v_proj(V_input)  # (b, k, E)

        # 2) Split in testine: (b, num_heads, seq_len, head_dim)
        Q = Q.view(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(b, k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(b, k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3) Scaled dot-product attention
        # attn_scores: (b, num_heads, n_q, k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (b, num_heads, n_q, k)
        attn_output = torch.matmul(attn_weights, V)    # (b, num_heads, n_q, head_dim)

        # 4) Concatenazione delle teste e proiezione finale
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (b, n_q, num_heads, head_dim)
        attn_output = attn_output.view(b, n_q, embed_dim)           # (b, n_q, E)
        attn_output = self.out_proj(attn_output)                    # (b, n_q, E)

        # 5) Residuo + Dropout + LayerNorm
        attn_output = self.dropout(attn_output)
        # Sommo l’input originale Q_input con l’output di attenzione
        output = self.norm(Q_input + attn_output)  # (b, n_q, E)

        return output
    
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        """
        One cross-attention block with:
          1) MultiHeadAttention (with its own residual + LayerNorm)
          2) Feed-forward network (FFN) + residual + LayerNorm

        Args:
            embed_dim (int): embedding dimension E
            num_heads (int): number of attention heads
            ffn_dim (int): hidden dimension inside the FFN (commonly 4*embed_dim)
            dropout (float): dropout probability for attention output and FFN output
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 1) Cross-attention module (uses your existing MultiHeadAttention)
        self.cross_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # LayerNorm after the first attention + residual
        self.norm1 = nn.LayerNorm(embed_dim)

        # 2) Feed-Forward Network: E → ffn_dim → E
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        # LayerNorm after the FFN + residual
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropouts for attention output and FFN output
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                Q_input: torch.Tensor,
                K_input: torch.Tensor,
                V_input: torch.Tensor,
                attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """
        Args:
            Q_input: (batch_size, n_q, E) — queries
            K_input: (batch_size, n_k, E) — keys
            V_input: (batch_size, n_k, E) — values
            attn_mask: optional mask applied inside attention
            key_padding_mask: optional padding mask (shape (batch_size, n_k))

        Returns:
            output (batch_size, n_q, E) after:
              - cross-attention + residual + LayerNorm
              - FFN + residual + LayerNorm
        """
        # --- Cross-Attention + Residual + LayerNorm ---
        # Your MultiHeadAttention already does: Q_proj, K_proj, V_proj, attention, output_proj, dropout, residual + norm.
        # Here we call it, then apply a second dropout + residual+norm if needed.
        # Actually, since your MultiHeadAttention ends with `output = LayerNorm(Q_input + attn_output)`,
        # you can treat its output as "A_res" directly.

        # Step 1: raw cross-attention (this applies its own residual + LayerNorm internally)
        A_res = self.cross_attn(Q_input, K_input, V_input)  # (b, n_q, E)

        # (Optional) If you want an extra dropout on the result:
        A_res = self.dropout1(A_res)

        # Note: if you strictly want to separate the residual from your MHA, you could modify MultiHeadAttention
        # to output only attn_output (pre-residual), then do residual+norm here. But since your MHA already does that,
        # we take A_res as the post-attention residual+norm result.

        # --- Feed-Forward + Residual + LayerNorm ---
        # Step 2: pass A_res through the FFN
        F_out = self.ffn(A_res)          # (b, n_q, E)
        F_out = self.dropout2(F_out)     # apply dropout after FFN

        # Step 3: second residual + LayerNorm
        output = self.norm2(A_res + F_out)  # (b, n_q, E)

        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float = 0.0):
        """
        Multi-head self-attention for a sequence of length n (e.g. flattened H×W).
        Args:
            embed_dim (int): embedding dimension E.
            num_heads (int): number of attention heads.
            dropout_p (float): dropout probability applied after out_proj.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout and LayerNorm
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: (batch_size, seq_len, embed_dim)  e.g. seq_len = H*W.
        Returns:
            out: (batch_size, seq_len, embed_dim)
        """
        b, n, E = x_flat.shape
        assert E == self.embed_dim, "Embedding dimension mismatch"

        # 1) Project inputs to Q, K, V
        Q = self.q_proj(x_flat)  # (b, n, E)
        K = self.k_proj(x_flat)  # (b, n, E)
        V = self.v_proj(x_flat)  # (b, n, E)

        # 2) Reshape for multi-head: (b, num_heads, n, head_dim)
        Q = Q.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3) Scaled dot-product attention per head
        # Flatten heads: treat each head as a separate example for the helper function
        Q_flat = Q.reshape(b * self.num_heads, n, self.head_dim)  # (b*heads, n, head_dim)
        K_flat = K.reshape(b * self.num_heads, n, self.head_dim)
        V_flat = V.reshape(b * self.num_heads, n, self.head_dim)

        attn_out = scaled_dot_product_attention(Q_flat, K_flat, V_flat,
                                                dropout_p=self.dropout.p,
                                                is_causal=False)  # (b*heads, n, head_dim)

        # 4) Restore shape: (b, heads, n, head_dim)
        attn_out = attn_out.view(b, self.num_heads, n, self.head_dim)

        # 5) Concatenate heads: (b, n, E)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(b, n, E)

        # 6) Final linear projection + dropout
        attn_out = self.out_proj(attn_out)  # (b, n, E)
        attn_out = self.dropout(attn_out)

        # 7) Residual + LayerNorm
        out = self.norm(x_flat + attn_out)  # (b, n, E)
        return out
    
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        """
        One full self-attention block (like a Transformer encoder layer):
          1) MultiHeadSelfAttention (with internal residual+LayerNorm)
          2) Feed-Forward Network (E -> ffn_dim -> E) + residual+LayerNorm

        Args:
            embed_dim (int): embedding dimension E
            num_heads (int): number of attention heads
            ffn_dim (int): hidden dimension of the FFN (commonly 4 * embed_dim)
            dropout (float): dropout probability after attention and FFN
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 1) Multi-Head Self-Attention (resid + norm built inside it)
        self.self_attn = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_p=dropout)

        # 2) Feed-Forward Network: E -> ffn_dim -> E
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)  # second LayerNorm after FFN
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, embed_dim)
        Returns:
            out: (batch_size, seq_len, embed_dim)
        """
        # 1) Self-Attention (includes its own residual + LayerNorm)
        attn_out = self.self_attn(x)  # shape: (batch, seq_len, embed_dim)

        # 2) Feed-Forward on top of attn_out
        ffn_out = self.ffn(attn_out)       # (batch, seq_len, embed_dim)
        ffn_out = self.dropout2(ffn_out)

        # 3) Second residual + LayerNorm
        out = self.norm2(attn_out + ffn_out)  # (batch, seq_len, embed_dim)
        return out
