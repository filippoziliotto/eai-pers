# Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from utils.utils import soft_argmax_coords


# TODO: delete the COORDINATEPREDICTION CLASS
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
        
        
    def forward(self, feature_map, query):
        """
        Processes the input map features and text query to predict coordinates.
        
        Args:
            map_features: Tensor of shape (b, w, h, E) where each spatial location is an embedding.
            query: A text query.
        """
        output = {}
        # This is the output of the MHA part
        b, w, h, E = feature_map.shape
        
        # Encode the query
        query = self.encode_query(query)
        assert query.shape == (b, E), "Query embedding must have shape (b, E)"
        
        # Process with simple cosine similarity
        if self.process_type in ["base"]:
            # Calculate cosine similarity
            value_map = self.cosine_similarity(
                feature_map, 
                query
            ) # b x w x h
            output["value_map"] = value_map.view(b, w, h, 1)
            
            # compute soft-argmax coords
            coords = soft_argmax_coords(value_map, self.tau)
            output["coords"] = coords
            return output  
        
        elif self.process_type in ["conv"]:
            # Apply a convolutional layer to the feature map
            feature_map = feature_map.permute(0, 3, 1, 2)
            feature_map = self.conv(feature_map)
            feature_map = self.conv2(feature_map)
            feature_map = feature_map.permute(0, 2, 3, 1)
            
            # Apply convolutions to query
            query = query.unsqueeze(1).unsqueeze(2)
            query = self.conv(query)
            query = self.conv2(query)
            query = query.squeeze(1).squeeze(2)
            
            # Calculate cosine similarity
            value_map = self.cosine_similarity(
                feature_map, 
                query
            ) # b x w x h
            output["value_map"] = value_map.view(b, w, h, 1)
            
            # compute soft-argmax coords
            coords = soft_argmax_coords(value_map, self.tau)
            output["coords"] = coords
            return output
        
    def encode_query(self, query):
        """
        Encodes a given query into an embedding.
        
        Args:
            query (str): The query to encode.
        
        Returns:
            dict: Contains a key 'text' with tensor shape (b, E) representing the query embedding.
        """
        return self.encoder.get_embeddings(text=query, modality='text')
        