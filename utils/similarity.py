import torch
import torch.nn.functional as F

def cosine_similarity(x: torch.Tensor, y: torch.Tensor, method='pytorch') -> torch.Tensor:
    """
    Compute cosine similarity between two tensors.
    
    Args:
        x (torch.Tensor): First input tensor of shape (w, h, E)
        y (torch.Tensor): Second input tensor of shape (1, E)
        
    Returns:
        torch.Tensor: Cosine similarity values of shape (w, h)
    """

    if method in ['scratch']:
        b, w, h, E = x.size()  # x: (b, w, h, E)
        b_y, E_y = y.size()  # y: (b, E)
        
        # Ensure batch sizes and embedding dimensions match
        assert b == b_y, "Batch sizes of x and y must match"
        assert E == E_y, "Embedding dimensions of x and y must match"
        
        # Reshape y to match dimensions for broadcasting: (b, 1, 1, E)
        y = y.view(b, 1, 1, E)
        
        # Compute dot product along the last dimension (embedding dimension)
        dot_product = torch.sum(x * y, dim=-1)  # (b, w, h)
        
        # Compute L2 norms for x and y along the embedding dimension
        x_norm = torch.norm(x, p=2, dim=-1)  # (b, w, h)
        y_norm = torch.norm(y, p=2, dim=-1)  # (b, 1, 1)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        
        # Compute cosine similarity
        similarity = dot_product / (x_norm * y_norm + eps)  # (b, w, h)
        
        # Reshape to have an extra channel dimension (b, w, h, 1)
        similarity = similarity.unsqueeze(-1)
            
    if method in ['pytorch']:
        b, w, h, E = x.size()  # x: (b, w, h, E)
        b_y, E_y = y.squeeze(1).size()  # y: (b, E)
        
        # Ensure batch sizes and embedding dimensions match
        assert b == b_y, "Batch sizes of x and y must match"
        assert E == E_y, "Embedding dimensions of x and y must match"
        
        # Reshape y to match dimensions for broadcasting: (b, 1, 1, E)
        y = y.view(b, 1, 1, E)
        
        # Compute cosine similarity along the last dimension (embedding dimension)
        similarity = F.cosine_similarity(x, y, dim=-1)  # similarity: (b, w, h)

    elif method in ['blip2_score']:
        # TODO
        pass
        
    else:
        raise ValueError(f"Invalid method: {method}")
    
    return similarity