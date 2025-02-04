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
        
        # Compute dot product
        dot_product = torch.sum(x * y, dim=-1)
        
        # Compute L2 norms
        x_norm = torch.norm(x, p=2, dim=-1)
        y_norm = torch.norm(y, p=2, dim=-1)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        
        # Compute cosine similarity
        similarity = dot_product / (x_norm * y_norm + eps)
        
    elif method in ['pytorch']:
        
        # Compute cosine similarity using PyTorch's F.cosine_similarity
        similarity = F.cosine_similarity(x, y.unsqueeze(0).unsqueeze(0), dim=-1)
        
    elif method in ['blip2_score']:
        # TODO
        pass
        
    else:
        raise ValueError(f"Invalid method: {method}")
    
    return similarity