
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

    def forward(self, map_features, query, loss_choice):
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
        
        if loss_choice == 'CE':
            return value_map.view(b, -1)  # Now shape: (batch, w*h)
            

        elif loss_choice in ['L1', 'L2']:
            # Step 2: Find the predicted coordinates (b, x', y') with max similarity
            # Here we choose to return only the max coordinate in a differentiable way.
            # You can adjust the temperature to control how "hard" the soft-argmax is.
            predicted_coords = []
            temperature = 0.1  # Lower values are closer to a hard argmax.
            for b in range(value_map.shape[0]):
                # Ensure that value_map[b] is a torch.Tensor with requires_grad (it should be if map_features/query_features are)
                coords = find_non_zero_neighborhood_indices(
                    value_map[b], w, neighborhood_size=self.pixels_per_meter//2, return_max=True, temperature=temperature
                )
                predicted_coords.append(coords)
            predicted_coords = torch.stack(predicted_coords)  # Shape: (b, 2)
            return predicted_coords