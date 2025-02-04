import torch
import torch.nn as nn
import torch.nn.functional as F

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
            Q_input: Tensor of shape (n_q, E) - Query input (e.g., map).
            K_input: Tensor of shape (k, E) - Key input (e.g., list of embeddings).
            V_input: Tensor of shape (k, E) - Value input (same shape as Key input).

        Returns:
            output: Tensor of shape (n_q, E)
        """
        n_q, embed_dim = Q_input.size()
        k, _ = K_input.size()
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"

        # Linear projections
        Q = self.q_proj(Q_input)  # (n_q, E)
        K = self.k_proj(K_input)  # (k, E)
        V = self.v_proj(V_input)  # (k, E)

        # Reshape for multi-head attention
        Q = Q.view(n_q, self.num_heads, self.head_dim).permute(1, 0, 2)  # (num_heads, n_q, head_dim)
        K = K.view(k, self.num_heads, self.head_dim).permute(1, 0, 2)    # (num_heads, k, head_dim)
        V = V.view(k, self.num_heads, self.head_dim).permute(1, 0, 2)    # (num_heads, k, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (num_heads, n_q, k)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (num_heads, n_q, k)
        attn_output = torch.matmul(attn_weights, V)    # (num_heads, n_q, head_dim)

        # Concatenate heads and project output
        attn_output = attn_output.permute(1, 0, 2).contiguous()  # (n_q, num_heads, head_dim)
        attn_output = attn_output.view(n_q, self.embed_dim)      # (n_q, E)
        attn_output = self.out_proj(attn_output)                # (n_q, E)

        return attn_output

# Example usage
if __name__ == "__main__":
    # Example dimensions
    n_q = 500 * 500  # Map size (e.g., w * h)
    k = 5    # Number of people
    E = 768   # Embedding dimension
    num_heads = 8  # Number of attention heads

    Q_input = torch.randn(n_q, E)  # Query input (map)
    K_input = torch.randn(k, E)   # Key input (list of embeddings)
    V_input = torch.randn(k, E)   # Value input (list of embeddings)

    mha = MultiHeadAttention(embed_dim=E, num_heads=num_heads)
    
    # Get number of parameters 
    print("Number of parameters:", sum(p.numel() for p in mha.parameters() if p.requires_grad))
    
    output = mha(Q_input, K_input, V_input)
    print("Output shape:", output.shape)  # Should be (n_q, E)

