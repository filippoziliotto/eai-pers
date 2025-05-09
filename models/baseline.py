import torch.nn as nn
import warnings

# Import custom models and configuration
from models.stages.second_stage import CoordinatePredictionModel

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)


class BaselineModel(nn.Module):
    def __init__(self, encoder, device):
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
        print("Initializing Baseline...")
        
        #TODO: add different baselines
        self.baseline_type = None

        # Initialize and move first and second stages to the specified device
        self.second_stage = CoordinatePredictionModel(encoder, method="hybrid").to(device)
        print("Baseline initialized.")
        
        if self.baseline_type:
            print(f"Using {self.baseline_type} baseline.")


    def forward(self, map_tensor, query):
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
        return self.second_stage(map_tensor, query)