# Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from utils.utils import calculate_dist_matrix


class PersonalizedFeatureMapper(nn.Module):
    def __init__(self, encoder, process_type="base", embed_dim=768, tau=1.0):
        """
        Args:
            encoder: An encoder with a method get_embeddings(text=..., modality='text') that returns
                     a dict with key 'text' and tensor shape (b, E).
            process_type: "base" or "conv", determines post-processing style.
            tau: Temperature for soft-argmax; lower -> sharper.
        """
        super().__init__()
        self.encoder = encoder
        self.process_type = process_type
        self.tau = tau
        
        # We'll use cosine similarity (over the embedding dimension) for the heatmap branch.
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
        
        if self.type in ["conv"]:
            # Apply a convolutional layer to the feature map
            self.conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=embed_dim // 2, out_channels=embed_dim // 4, kernel_size=3, padding=1)
        
        
    def forward(self, feature_map, query, gt_coords):
        """
        Processes the input map features and text query to predict coordinates.
        
        Args:
            map_features: Tensor of shape (b, w, h, E) where each spatial location is an embedding.
            query: A text query.
        """
        output = {}
        # This is the output of the MHA part
        b, h, w, E = feature_map.shape
        
        # Encode the query
        query = self.encode_query(query)
        query = query["text"]
        assert query.shape == (b, E), "Query embedding must have shape (b, E)"
        
        # Process with simple cosine similarity
        if self.process_type in ["base"]:

            # Expand query to match the spatial dimensions
            query_expanded = query.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, E)
            query_expanded = query_expanded.expand(-1, h, w, -1)  # (b, h, w, E)            

            # Calculate cosine similarity
            value_map = self.cosine_similarity(
                feature_map, 
                query_expanded
            ).view(b, h, w, 1) # b x h x w x 1
            output["value_map"] = value_map
            
            # compute soft-argmax coords
            dist_matrix = calculate_dist_matrix(value_map, gt_coords)
            output["dist_matrix"] = dist_matrix
            return output  
        
        else:
            raise ValueError(f"Unsupported process type: {self.process_type}. Supported types are 'base' and 'conv'.")
        
    def encode_query(self, query):
        """
        Encodes a given query into an embedding.
        
        Args:
            query (str): The query to encode.
        
        Returns:
            dict: Contains a key 'text' with tensor shape (b, E) representing the query embedding.
        """
        return self.encoder.get_embeddings(text=query, modality='text')
        