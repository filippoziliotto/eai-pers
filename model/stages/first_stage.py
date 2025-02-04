# Importing necessary libraries
import torch
import torch.nn as nn

# Importing custom models
from models.encoder import Blip2Encoder
from models.extractor import Extractor

# Importing utility functions
from utils.attention import MultiHeadAttention
from utils.utils import reshape_map

class MapAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder):
        """
        Initializes the MapAttentionModel with the given parameters.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            device (torch.device): The device to run the model on.
        """
        super(MapAttentionModel, self).__init__()
        self.encoder = encoder
        self.extractor = Extractor()
        self.mh_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def encode_descriptions(self, description):
        """
        Encodes a given description into embeddings.

        Args:
            description (str): The description to encode.

        Returns:
            torch.Tensor: The concatenated embeddings of the descriptions.
        """
        descriptions = self.extractor.separate(description)
        encoded_descriptions = [self.encoder.get_embeddings(desc, type='text') for desc in descriptions]
        attached_embs = torch.cat(encoded_descriptions, dim=0)
        return attached_embs

    def map_shaping(self, map_tensor):
        """
        Reshapes the map tensor.

        Args:
            map_tensor (torch.Tensor): The map tensor to reshape.

        Returns:
            torch.Tensor: The reshaped map tensor.
        """
        return reshape_map(map_tensor)

    def forward(self, map_tensor, description):
        """
        Forward pass of the model.

        Args:
            map_tensor (torch.Tensor): The map tensor.
            description (str): The description to encode.

        Returns:
            torch.Tensor: The output of the multi-head attention mechanism.
        """
        
        # Get description embeddings (k, E)
        desc_embeds = self.encode_descriptions(description)
        
        # Get reshaped map tensor (h*w, E)
        reshaped_map = self.map_shaping(map_tensor)
        
        output = self.mh_attention(reshaped_map, desc_embeds)
        return output.reshape(map_tensor.shape[0], map_tensor.shape[1], -1)