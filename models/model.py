import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import warnings

# Import custom models and configuration
from models.stages.first_stage import MapAttentionModel
from models.stages.second_stage import CoordinatePredictionModel
import config

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)


class RetrievalMapModel(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder, device):
        """
        Initializes the RetrievalMapModel.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            encoder (Blip2Encoder): The encoder model.
            pixels_per_meter (int): Map resolution in pixels per meter.
            use_scale_similarity (bool): Flag for using scale similarity.
            use_mlp_predictor (bool): Flag to include the MLP coordinate predictor.
            mlp_embed_dim (int): The embedding dimension for the MLP predictor.
            device (str): The device for model computation.
        """
        super().__init__()
        print("Initializing Model...")

        # Initialize and move first and second stages to the specified device
        self.first_stage = MapAttentionModel(embed_dim, num_heads, encoder).to(device)
        self.second_stage = CoordinatePredictionModel(encoder, method="hybrid").to(device)
        
        print("Model initialized.")

        # Display baseline evaluation message if applicable
        if config.VLFM_BASELINE:
            print("Evaluating VLFM Baseline...")

    def forward(self, description, map_tensor, query):
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
        # If VLFM baseline is active, compute similarity directly
        if config.VLFM_BASELINE:
            return self.second_stage(map_tensor, query)

        # Encode the map description using the first stage.
        if config.USE_GRAD_CHECK:
            dummy = torch.ones(1, device=map_tensor.device, requires_grad=True)
            embed_map = checkpoint.checkpoint(lambda m, d: self.first_stage(m, description), map_tensor, dummy)
        else:
            embed_map = self.first_stage(map_tensor, description)

        # Compute the similarity map from the second stage, based on the encoded map.
        return self.second_stage(embed_map, query)