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
            description (str): The description to encode. (b, k)

        Returns:
            torch.Tensor: The concatenated embeddings of the descriptions.
        """
            
        descriptions = self.extractor.separate(description)
        encoded_descriptions = [self.encoder.get_embeddings(text=desc, modality='text') for desc in descriptions]

        # Extract the 'text' tensors from each dictionary and concatenate them
        text_tensors = [desc['text'].unsqueeze(0) for desc in encoded_descriptions]
        # Attach embeddings as a batch (b, k, E), where k is batched to the longest element
        attached_embs = torch.cat(text_tensors, dim=0)
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
        
        # Get description embeddings (b, k, E)
        desc_embeds = self.encode_descriptions(description)
        
        # Get reshaped map tensor (b, h*w, E)
        reshaped_map = self.map_shaping(map_tensor)
        
        output = self.mh_attention(reshaped_map, desc_embeds, desc_embeds)
        return output.view(*map_tensor.shape[:3], -1)