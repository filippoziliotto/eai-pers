# Importing necessary libraries
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class ZeroShotCosineModel(nn.Module):
    def __init__(self, encoder):
        """
        Initializes the MapAttentionModel with the given parameters.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            device (torch.device): The device to run the model on.
        """
        super(ZeroShotCosineModel, self).__init__()
        self.encoder = encoder

    def encode_descriptions(self, descriptions):
        """
        descriptions: List[List[str]] of shape (batch_size, variable_lengths)

        Returns:
            Tensor of shape (batch_size, max_seq_len, embedding_dim)
        """
        # 1) Encode each list of strings into a tensor of shape (seq_len_i, E)
        embedding_tensors = []
        for desc_list in descriptions:
            emb_dict = self.encoder.get_embeddings(text=desc_list, modality='text')
            # emb_dict['text'] is (seq_len_i, E)
            embedding_tensors.append(emb_dict['text'])

        # 2) Pad them into a single tensor of shape (batch, max_seq, E), zero‐padding shorter ones
        #    pad_sequence defaults to padding with zeros.
        padded: torch.Tensor = pad_sequence(
            embedding_tensors,
            batch_first=True,  # → (batch, max_seq_len, E)
            padding_value=0.0
        )

        return padded
    
    def forward(self, feature_map, query_tensor, description_tensor, top_k=1):
        # Normalize feature map and description embeddings
        feat_norm = F.normalize(feature_map, p=2, dim=-1)
        desc_norm = F.normalize(description_tensor, p=2, dim=-1)
        
        # Flatten the spatial dimensions of the feature map
        b, H, W, E = feat_norm.shape
        _, K, _ = desc_norm.shape
        feat_flat = feat_norm.view(b, H * W, E)
        
        # Transpose description tensor for batch matrix multiplication
        desc_t = desc_norm.transpose(1, 2)
        
        # Compute cosine similarity between each spatial location and each description
        desc_value_flat = torch.bmm(feat_flat, desc_t)
        
        # Reshape similarity scores to (b, H, W, K)
        desc_value_map = desc_value_flat.view(b, H, W, K)
        
        # Flatten similarity map for masking
        desc_flat = desc_value_map.view(b, H * W, K)
        
        # Find the index of the maximum similarity for each description
        _, max_idxs = desc_flat.max(dim=1)
        
        # Build a boolean mask that is True only at the top-K indices
        mask = torch.zeros((b, H * W), dtype=torch.bool, device=feature_map.device)
        mask = mask.scatter_(1, max_idxs, True)
        
        # Reshape mask to (b, H, W, 1)
        mask = mask.view(b, H, W, 1)
        
        # Apply mask to the feature map (zero out all but the top-K positions)
        feature_map = feature_map * mask
        
        # Prepare query tensor for cosine computation
        query_tensor = query_tensor.unsqueeze(1).unsqueeze(2)
        query_tensor = query_tensor.expand(-1, H, W, -1)
        
        # Compute cosine similarity between masked feature map and query
        value_map = self.cosine_similarity(feature_map, query_tensor)
        
        # Extract the maximum similarity value and its spatial coordinates
        max_value_map, max_index = value_map.view(b, -1).max(dim=-1)
        
        return max_index, max_value_map
    
    def cosine_similarity(self, x, y):
        """
        Computes the cosine similarity between two tensors.
        
        Args:
            x (Tensor): First tensor.
            y (Tensor): Second tensor.
        
        Returns:
            Tensor: Cosine similarity between x and y.
        """
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)
    