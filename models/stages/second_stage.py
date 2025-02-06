
# Library imports
import torch
import torch.nn as nn
import numpy as np

# Local imports
from utils.similarity import cosine_similarity
from utils.utils import find_non_zero_neighborhood_indices


class SimilarityMapModel(nn.Module):
    def __init__(self, encoder, cosine_method, pixels_per_meter):
        """
        Computes the loss based on cosine similarity between a map and a query vector.
        """
        super(SimilarityMapModel, self).__init__()
        self.cosine_method = cosine_method
        self.encoder = encoder
        self.pixels_per_meter = pixels_per_meter
        
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
            loss: Computed loss value.
            predicted_coords: Tuple (x', y') - Predicted coordinates from the similarity map.
        """
        b, w, h, E = map_features.size()
        query_features = self.encode_query(query)    
        query_features = query_features['text']
        
        assert query_features.size() == (b, E), "Query feature must have shape (b, E)"
        
        # Step 1: Compute cosine similarity for each pixel (wi, hj, E) and query (1, E)
        value_map = cosine_similarity(
            map_features, query_features, method=self.cosine_method
        )  # Shape: (b, w, h)

        # Step 2: Find the predicted coordinates (b, x', y') with max similarity
        predicted_coords = []
        for b in range(value_map.shape[0]):
            coords = find_non_zero_neighborhood_indices(value_map[b], w, neighborhood_size=self.pixels_per_meter//2, return_max=True)
            predicted_coords.append(coords)
        predicted_coords = torch.tensor(predicted_coords, device=value_map.device)

        return predicted_coords