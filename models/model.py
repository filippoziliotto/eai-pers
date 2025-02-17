# Importing necessary libraries
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch

# Importing custom models
from models.stages.first_stage import MapAttentionModel
from models.stages.second_stage import SimilarityMapModel

# Import config
import config

# Warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class RetrievalMapModel(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder, pixels_per_meter, use_scale_similarity, device):
        """
        Initializes the RetrievalMapModel.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            encoder (Blip2Encoder): The encoder model.
            pixels_per_meter (int): The resolution of the map in pixels per meter.
            device (str): The device for model computation.
            use_vlfm_baseline (bool): Flag to use the VLFM baseline approach.
        """
        super(RetrievalMapModel, self).__init__()
        print("Initializing Model...")
        
        # Initialize first stage and move to device
        self.first_stage = MapAttentionModel(embed_dim, num_heads, encoder).to(device)
        # Initialize second stage and move to device
        self.second_stage = SimilarityMapModel(encoder, pixels_per_meter, use_scale_similarity).to(device)
        print("Model initialized.")

        # If using VLFM baseline, print a message
        if config.VLFM_BASELINE:
            print("Evaluating VLFM Baseline...")

    def forward(self, description, map_tensor, query):
        """
        Args:
            description (str): The description of the map.
            map_tensor: Tensor of shape (w, h, C) - The map tensor.
            query (str): The query to find in the map.
            loss_choice (str): The loss function choice.

        Returns:
            predicted_coords: Tuple (x', y') - Predicted coordinates from the similarity map.
        """
        
        # If using VLFM baseline, skip the first stage and directly compute similarity
        # bewteen the original feature map and the query
        if config.VLFM_BASELINE:
            return self.second_stage(map_tensor, query)
        
        # Step 1: Encode the description
        if config.USE_GRAD_CHECK:
            # Use gradient checkpointing for the heavy first stage.
            # We use a lambda to "capture" the non-tensor argument `description`.
            # Create a dummy tensor that requires grad
            dummy = torch.ones(1, device=map_tensor.device, requires_grad=True)
            embed_map = checkpoint.checkpoint(lambda m, d: self.first_stage(m, description), map_tensor, dummy)
        else:
            embed_map = self.first_stage(map_tensor, description)

        # Step 2: Encode the query
        return self.second_stage(embed_map, query)