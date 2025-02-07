# Importing necessary libraries
import torch
import torch.nn as nn

# Importing custom models
from models.stages.first_stage import MapAttentionModel
from models.stages.second_stage import SimilarityMapModel


class RetrievalMapModel(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder, cosine_method, pixels_per_meter, device):
        """
        Initializes the RetrievalMapModel with the given parameters.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            encoder (Blip2Encoder): The encoder model.
            cosine_method (str): The cosine method to use.
            pixels_per_meter (int): The number of pixels per meter.
        """
        super(RetrievalMapModel, self).__init__()
        print("Initializing Model...")
        self.first_stage = MapAttentionModel(embed_dim, num_heads, encoder).to(device)
        self.second_stage = SimilarityMapModel(encoder, cosine_method, pixels_per_meter).to(device)
        print("Model initialized.")

    def forward(self, description, map_tensor, query, loss_choice):
        """
        Args:
            description (str): The description of the map.
            map_tensor: Tensor of shape (w, h, C) - The map tensor.
            query (str): The query to find in the map.

        Returns:
            predicted_coords: Tuple (x', y') - Predicted coordinates from the similarity map.
        """
        # Step 1: Encode the description
        embed_map = self.first_stage(map_tensor, description)

        # Step 2: Encode the query
        predicted_coords = self.second_stage(embed_map, query, loss_choice)

        return predicted_coords