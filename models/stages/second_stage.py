
# Library imports
import torch
import torch.nn as nn
import numpy as np

class SimilarityMapModel(nn.Module):
    def __init__(self, encoder, pixels_per_meter):
        """
        Computes the loss based on cosine similarity between a map and a query vector.
        """
        super(SimilarityMapModel, self).__init__()
        # Initialize the encoder
        self.encoder = encoder
        self.pixels_per_meter = pixels_per_meter
        
        # Instatiate the similarity
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-08)
        
    def encode_query(self, query):
        """
        Encodes a given query into an embedding.

        Args:
            query (str): The query to encode.

        Returns:
            torch.Tensor: The encoded query.
        """
        return self.encoder.get_embeddings(text=query, modality='text')

    def forward(self, map_features, query):
        """
        Args:
            map_features: Tensor of shape (b, w, h, E) - Feature map from the previous class.
            query_feature: Tensor of shape (b, 1, E) - Query vector for comparison.
            ground_truth_coords: Tuple (b, x, y) - Ground truth pixel coordinates.
        
        Returns:
            similarity: Tensor of shape (b, w, h) - Similarity map.
        """
        b, w, h, E = map_features.size()
        query_features = self.encode_query(query)    
        query_features = query_features['text']
        assert query_features.size() == (b, E), "Query feature must have shape (b, E)"
        
        # Reshape y to match dimensions for broadcasting: (b, 1, 1, E)
        x = map_features.view(b, w, h, E)
        y = query_features.view(b, 1, 1, E)
        
        # Compute cosine similarity along the last dimension (embedding dimension)
        return self.cosine_similarity(x, y)  # similarity: (b, w, h)

