import torch
import torch.nn as nn

from utils.similarity import cosine_similarity
from utils.utils import find_non_zero_neighborhood


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
        return self.encoder.get_embeddings(query, type='text')

    def forward(self, map_features, query):
        """
        Args:
            map_features: Tensor of shape (w, h, E) - Feature map from the previous class.
            query_feature: Tensor of shape (1, E) - Query vector for comparison.
            ground_truth_coords: Tuple (x, y) - Ground truth pixel coordinates.
        
        Returns:
            loss: Computed loss value.
            predicted_coords: Tuple (x', y') - Predicted coordinates from the similarity map.
        """
        w, h, E = map_features.size()
        query_feature = self.encode_query(query)    
        assert query_feature.size() == (1, E), "Query feature must have shape (1, E)"
        
        # Step 1: Compute cosine similarity for each pixel (wi, hj, E) and query (1, E)
        value_map = cosine_similarity(
            map_features, query_feature, method=self.cosine_method
        )  # Shape: (w, h)

        # Step 2: Find the predicted coordinates (x', y') with max similarity
        predicted_coords = find_non_zero_neighborhood(value_map, w, neighborhood_size=self.pixels_per_meter)

        return predicted_coords