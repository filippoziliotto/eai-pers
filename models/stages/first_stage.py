# Importing necessary libraries
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Importing utility functions
from utils.attention import CrossAttentionBlock, SelfAttentionBlock
from utils.positional import Learnable2DPositionalEncodingMax

class MapAttentionModel(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 ffn_dim,
                 encoder, 
                 num_cross_layers: int = 2,
                 num_self_layers: int = 1,
                 dropout: float = 0.1,
                 use_self_attention: bool = False, 
                 use_pos_embed: bool = True):
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
        
        if ffn_dim is None:
            ffn_dim = 2 * embed_dim
        
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_cross_layers)
        ])
        
        self.use_self_attention = use_self_attention
        # Now build a stack of SelfAttentionBlock if requested
        if self.use_self_attention:
            self.self_blocks = nn.ModuleList([
                SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
                for _ in range(num_self_layers)
            ])

        # Positional encoding learnable
        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
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
    
    def forward(self, feature_map: torch.Tensor, descriptions: list) -> torch.Tensor:
        b, H, W, E = feature_map.shape
        # 1) encode descriptions → desc_embeds (b, k, E)
        desc_embeds = self.encode_descriptions(descriptions)

        # 2) optional pos-emb
        if self.use_pos_embed:
            pe_hw = self.pos_enc(H, W)  # (H, W, E)
            pe_hw = pe_hw.unsqueeze(0).expand(b, -1, -1, -1)  # (b, H, W, E)
            fmap_pe = feature_map + pe_hw
        else:
            fmap_pe = feature_map

        # 3) flatten to (b, H*W, E) for cross-attention
        Q = fmap_pe.view(b, H * W, E)

        # 4) stack cross-attention blocks
        for cross_block in self.cross_blocks:
            Q = cross_block(Q, desc_embeds, desc_embeds)

        # 5) optionally reshape back to (b, H, W, E) or keep (b, H*W, E)
        #    If you want to apply self-attention on the *flattened* map, keep it flat:
        if self.use_self_attention:
            for self_block in self.self_blocks:
                Q = self_block(Q)  # Q remains shape (b, H*W, E)

        # 6) reshape back before returning
        output_map = Q.view(b, H, W, E)
        return output_map