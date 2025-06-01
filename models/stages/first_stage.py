# Importing necessary libraries
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Importing utility functions
from utils.attention import MultiHeadAttention, MultiHeadSelfAttention
from utils.utils import reshape_map
from utils.positional import Learnable2DPositionalEncodingMax

class MapAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder, use_self_attention=False, use_pos_embed=True):
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

        # Positional encoding learnable
        if use_pos_embed:
            # Use learnable 2D positional encoding
            self.pos_enc = Learnable2DPositionalEncodingMax(E=embed_dim)


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
        
        b, H, W, E = feature_map.shape
        # 1) Get desc_embeds as before
        desc_embeds = self.encode_descriptions(description)  # (b, k, E)

        # 2) Get the positional map for (H, W)
        pe_hw = self.pos_enc(H, W)               # (H, W, E)
        pe_hw = pe_hw.unsqueeze(0).expand(b, -1, -1, -1)  # (b, H, W, E)

        # 3) Add to feature_map and reshape
        fmap_pe = feature_map + pe_hw            # (b, H, W, E)
        reshaped_map = fmap_pe.view(b, H*W, E)   # (b, H*W, E)

        # 4) Cross-attention
        output = self.mh_attention(reshaped_map, desc_embeds, desc_embeds)  # (b, H*W, E)

        # 5) (If desired) self-attention
        if self.use_self_attention:
            output = self.mh_sattention(output)

        # 6) Reshape to (b, H, W, E) and return
        return output.view(b, H, W, E)