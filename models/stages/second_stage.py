
# Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityMapModel(nn.Module):
    def __init__(self, encoder, pixels_per_meter, use_scale_similarity=False):
        """
        Computes the loss based on cosine similarity between a map and a query vector.
        """
        super(SimilarityMapModel, self).__init__()
        # Initialize the encoder
        self.encoder = encoder
        self.pixels_per_meter = pixels_per_meter
        
        # Instatiate the similarity
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-08)
        
        # Use a trainable scale parameters for similarity map amplification
        self.use_scale_similarity = use_scale_similarity
        if self.use_scale_similarity:
            # Small CNN module to predict per-pixel scaling
            self.scale_predictor = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
            )
        else:
            self.scale_predictor = 1.
        
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
        
        if not self.use_scale_similarity:
            return self.cosine_similarity(x, y) # (b, w, h)
        
        # Compute cosine similarity between x and y
        # Add a channel dimension before feeding into the scale predictor
        similarity_map = self.cosine_similarity(x, y).unsqueeze(1)  # shape: (b, 1, w, h)

        # Generate the scale map and remove the added channel
        scale_map = self.scale_predictor(similarity_map).squeeze(1)  # shape: (b, w, h)

        # Return the element-wise product of the scale map and the initial cosine similarity
        return scale_map * similarity_map.squeeze(1)

