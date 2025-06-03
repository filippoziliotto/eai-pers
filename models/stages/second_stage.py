# Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from utils.utils import soft_argmax_coords
from utils.similarity import LearnableScalarSimilarity


class PersonalizedFeatureMapper(nn.Module):
    def __init__(self, encoder, process_type="base", embed_dim=768, tau=1.0):
        """
        Args:
            encoder: An encoder with a method get_embeddings(text=..., modality='text') that returns
                     a dict with key 'text' and tensor shape (b, E).
            process_type: "base" or "conv", determines post-processing style.
            tau: Temperature for soft-argmax; lower -> sharper.
        """
        super().__init__()
        self.encoder = encoder
        self.process_type = process_type
        self.tau = tau
        
        if self.process_type == "base":
            # Use unbounded learned‐scalar similarity (E → 1)
            self.learnable_sim = LearnableScalarSimilarity(input_dim=embed_dim)

        elif self.process_type == "conv":
            # Two convolutional layers to reduce E → E/4 (must be divisible by 4)
            self.conv  = nn.Conv2d(in_channels=embed_dim,   out_channels=embed_dim // 2, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=embed_dim // 2, out_channels=embed_dim // 4, kernel_size=3, padding=1)
            # Then use LearnableScalarSimilarity on reduced (E/4 → 1)
            self.learnable_sim = LearnableScalarSimilarity(input_dim=embed_dim // 4)

        else:
            raise ValueError(f"Unknown process_type: {self.process_type!r}")
        
        
    def forward(self, feature_map, query, gt_coords=None):
        """
        Processes the input map features and text query to predict coordinates.
        
        Args:
            map_features: Tensor of shape (b, w, h, E) where each spatial location is an embedding.
            query: A text query.
        """
        output = {}
        # This is the output of the MHA part
        b, h, w, E = feature_map.shape

        # Encode the query
        # 1) Encode the query to (b, E)
        query_dict = self.encode_query(query)
        q_emb = query_dict["text"]
        assert q_emb.shape == (b, E), "Query embedding must have shape (b, E)"
        

        if self.process_type == "base":
            # Directly compute unbounded learned score map:
            # sim_map: (b, h, w)
            sim_map = self.learnable_sim(feature_map, q_emb)
            # Add a trailing singleton channel to match (b, h, w, 1)
            value_map = sim_map.view(b, h, w, 1)
            output["value_map"] = value_map

            # 2) Soft-argmax → (b, 2)
            coords = soft_argmax_coords(value_map, self.tau)
            output["coords"] = coords
            return output

        elif self.process_type == "conv":
            # 1) Apply convolutions to feature_map:
            #    Input: (b, H, W, E) → permute to (b, E, H, W)
            x = feature_map.permute(0, 3, 1, 2)  # → (b, E, h, w)
            x = self.conv(x)                     # → (b, E/2, h, w)
            x = self.conv2(x)                    # → (b, E/4, h, w)
            # Permute back to (b, h, w, E/4)
            x = x.permute(0, 2, 3, 1)            # → (b, h, w, E/4)

            # 2) Apply the same two convs to the query embedding:
            #    q_emb: (b, E) → unsqueeze → (b, E, 1, 1) → conv → conv2 → (b, E/4, 1, 1)
            q = q_emb.unsqueeze(-1).unsqueeze(-1).permute(0, 3, 1, 2)  # → (b, E, 1, 1)
            q = self.conv(q)                                          # → (b, E/2, 1, 1)
            q = self.conv2(q)                                         # → (b, E/4, 1, 1)
            q = q.squeeze(-1).squeeze(-1)                             # → (b, E/4)

            # 3) Compute unbounded learned score on reduced embeddings:
            #    x: (b, h, w, E/4), q: (b, E/4) → sim_map (b, h, w)
            sim_map = self.learnable_sim(x, q)
            value_map = sim_map.view(b, h, w, 1)
            output["value_map"] = value_map

            # 4) Soft-argmax → (b, 2)
            coords = soft_argmax_coords(value_map, self.tau)
            output["coords"] = coords
            return output
        
    def encode_query(self, query):
        """
        Encodes a given query into an embedding.
        
        Args:
            query (str): The query to encode.
        
        Returns:
            dict: Contains a key 'text' with tensor shape (b, E) representing the query embedding.
        """
        return self.encoder.get_embeddings(text=query, modality='text')
        