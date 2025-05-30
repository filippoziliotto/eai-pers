# Importing necessary libraries
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Importing utility functions
from utils.attention import MultiHeadAttention, MultiHeadSelfAttention
from utils.utils import reshape_map

class MapAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder, use_self_attention=False):
        """
        Initializes the MapAttentionModel with the given parameters.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            encoder (Blip2Encoder): The encoder model used for text embeddings.
            use_self_attention (bool): Whether to use self-attention in the model.
        """
        super(MapAttentionModel, self).__init__()
        self.encoder = encoder
        self.mh_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        self.use_self_attention = use_self_attention
        if self.use_self_attention:
            # Apply self-attention if specified
            self.mh_sattention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)

    def encode_descriptions(self, descriptions):
        """
        descriptions: List[List[str]] of shape (batch_size, variable_lengths)

        Returns:
            Tensor of shape (batch_size, max_seq_len, embedding_dim)
        """
        # 1) Encode each list of strings into a tensor of shape (seq_len_i, E)
        embedding_tensors = []
        for desc_list in descriptions:
            emb_dict = self.encoder.get_embeddings(text=desc_list, modality='text')
            # emb_dict['text'] is (seq_len_i, E)
            embedding_tensors.append(emb_dict['text'])

        # 2) Pad them into a single tensor of shape (batch, max_seq, E), zero‐padding shorter ones
        #    pad_sequence defaults to padding with zeros.
        padded: torch.Tensor = pad_sequence(
            embedding_tensors,
            batch_first=True,  # → (batch, max_seq_len, E)
            padding_value=0.0
        )

        return padded
    
    def forward(self, feature_map, description):
        """
        Forward pass of the model.

        Args:
            feature_map (torch.Tensor): The map tensor.
            description (str): The description to encode.

        Returns:
            torch.Tensor: The output of the multi-head attention mechanism.
        """
        
        # Get description embeddings (b, k, E)
        desc_embeds = self.encode_descriptions(description)
        
        # Get reshaped map tensor (b, h*w, E)
        reshaped_map = reshape_map(feature_map)
        
        # Cross-attention
        output = self.mh_attention(reshaped_map, desc_embeds, desc_embeds)
        
        # Self-attention if specified
        if self.use_self_attention:
            output = self.mh_sattention(output)
        

        return output.view(*feature_map.shape[:3], -1)