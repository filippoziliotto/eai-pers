import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Simplified multi-head attention class.

        Args:
            embed_dim (int): The embedding dimension (E).
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
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

    def forward(self, Q_input, K_input, V_input):
        """
        Multi-head attention forward pass.

        Args:
            Q_input: Tensor of shape (b, n_q, E) - Query input (e.g., map).
            K_input: Tensor of shape (b, k, E) - Key input (e.g., list of embeddings).
            V_input: Tensor of shape (b, k, E) - Value input (same shape as Key input).

        Returns:
            output: Tensor of shape (b, n_q, E)
        """
        b, n_q, embed_dim = Q_input.size()
        b, k, _ = K_input.size()
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"

        # Linear projections
        Q = self.q_proj(Q_input)  # (b, n_q, E)
        K = self.k_proj(K_input)  # (b, k, E)
        V = self.v_proj(V_input)  # (b, k, E)

        # Reshape for multi-head attention
        Q = Q.view(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (b, num_heads, n_q, head_dim)
        K = K.view(b, k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)    # (b, num_heads, k, head_dim)
        V = V.view(b, k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)    # (b, num_heads, k, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (b, num_heads, n_q, k)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (b, num_heads, n_q, k)
        attn_output = torch.matmul(attn_weights, V)    # (b, num_heads, n_q, head_dim)

        # Concatenate heads and project output
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (b, n_q, num_heads, head_dim)
        attn_output = attn_output.view(b, n_q, self.embed_dim)      # (b, n_q, E)
        attn_output = self.out_proj(attn_output)                    # (b, n_q, E)

        return attn_output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.0):
        """
        Multi-head self-attention for 2D spatial data (e.g., feature maps).

        Args:
            embed_dim (int): The embedding dimension (E).
            num_heads (int): The number of attention heads.
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
        
        # Dropout layer
        self.dropout = dropout_p

    def forward(self, x_flat):
        """
        Args:
            x: Tensor of shape (b, H*W, E)

        Returns:
            out: Tensor of shape (b, H*W, E)
        """
        b, n, E = x_flat.shape
        assert E == self.embed_dim, "Embedding dimension mismatch"
        
        # Linear projections
        Q = self.q_proj(x_flat)  # (b, n, E)
        K = self.k_proj(x_flat)  # (b, n, E)
        V = self.v_proj(x_flat)  # (b, n, E)
        
        # Use scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout, is_causal=False)

        # Project output
        out = self.out_proj(attn_output)  # (b, n, E)
        
        return out
