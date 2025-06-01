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

class MultiHeadSelfAttention(nn.Module):
    def init(self, embed_dim, num_heads, dropout_p=0.0):
        """
            Multi-head self-attention for 2D spatial data (e.g., flattened feature maps).

            Args:
                embed_dim (int): The embedding dimension (E).
                num_heads (int): The number of attention heads.
                dropout_p (float): Dropout probability.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout and LayerNorm
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_flat):
        """
        Args:
            x_flat: Tensor of shape (b, H*W, E)

        Returns:
            out: Tensor of shape (b, H*W, E)
        """
        b, n, E = x_flat.shape
        assert E == self.embed_dim, "Embedding dimension mismatch"

        # Project inputs to Q, K, V
        Q = self.q_proj(x_flat)  # (b, n, E)
        K = self.k_proj(x_flat)  # (b, n, E)
        V = self.v_proj(x_flat)  # (b, n, E)

        # Compute attention
        attn_output = scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout.p, is_causal=False)  # (b, n, E)

        # Final projection
        attn_output = self.out_proj(attn_output)  # (b, n, E)
        attn_output = self.dropout(attn_output)

        # Residual connection + LayerNorm
        out = self.norm(x_flat + attn_output)  # (b, n, E)

        return out