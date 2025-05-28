import torch.nn as nn
import warnings

# Import custom models and configuration
from models.stages.first_stage import MapAttentionModel
from models.stages.second_stage import PersonalizedFeatureMapper

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)


class RetrievalMapModel(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder, type, tau, device):
        """
        Initializes the RetrievalMapModel.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            encoder (Blip2Encoder): The encoder model.
            device (str): The device for model computation.
        """
        super().__init__()
        print("Initializing Model...")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.process_type = type
        self.tau = tau

        # Initialize and move first and second stages to the specified device
        self.first_stage = MapAttentionModel(self.embed_dim, 
                                             self.num_heads, 
                                             encoder
                                             ).to(self.device)
        self.second_stage = PersonalizedFeatureMapper(encoder, 
                                                      process_type=self.process_type, 
                                                      embed_dim=self.embed_dim, 
                                                      tau=self.tau
                                                      ).to(self.device)
        
        print("Model initialized.")

    def forward(self, description, map_tensor, query, gt_coords):
        """
        Performs a forward pass through the model.

        Args:
            description (str): Description of the map.
            map_tensor (Tensor): The map tensor of shape (w, h, C).
            query (str): The query to locate in the map.

        Returns:
            Either:
              - The similarity map result as computed by second_stage if not using MLP predictor.
              - A tuple of predicted coordinates from the MLP predictor.
        """

        # Encode the map description using the first stage.
        embed_map = self.first_stage(map_tensor, description)

        # Compute the similarity map from the second stage, based on the encoded map.
        return self.second_stage(embed_map, query, gt_coords)