
# Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinatePredictionModel(nn.Module):
    def __init__(self, encoder, pixels_per_meter, method="global", hidden_dim=128):
        """
        A coordinate prediction model that can operate in two modes:
        
        1. 'global'  - Global Pooling + FC Regression.
        2. 'hybrid'  - Hybrid Multi-Task: regression branch plus heatmap prediction (via soft-argmax).
        
        Args:
            encoder: An encoder with a method get_embeddings(text=..., modality='text') that returns
                     a dict with key 'text' and tensor shape (b, E). It is assumed that encoder.embedding_dim exists.
            pixels_per_meter (float): Scaling factor from pixel space to physical units (if needed).
            method (str): Either "global" or "hybrid".
            hidden_dim (int): Hidden layer dimension for the regression MLP.
        """
        super(CoordinatePredictionModel, self).__init__()
        self.encoder = encoder
        self.pixels_per_meter = pixels_per_meter
        self.method = method
        
        # We'll use cosine similarity (over the embedding dimension) for the heatmap branch.
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
        
        # --- Global Regression Branch (Strategy 1) ---
        # Expecting map_features of shape (b, w, h, E), we first permute to (b, E, w, h)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # The encoder is assumed to produce embeddings of size encoder.embedding_dim.
        feat_dim = encoder.encoder.itm_head.in_features
        # We'll concatenate the pooled map feature (shape: (b, feat_dim)) with the query embedding (b, feat_dim)
        # and then use an MLP to regress (x,y).
        self.regressor = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        if self.method == "hybrid":
            # --- Heatmap Branch (part of Strategy 5) ---
            # Start by computing a cosine similarity map between map_features and query.
            # We add a small conv head (similar to your original scale predictor) to refine this similarity.
            self.heatmap_conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            )
        
    def encode_query(self, query):
        """
        Encodes a given query into an embedding.
        
        Args:
            query (str): The query to encode.
        
        Returns:
            dict: Must contain a key 'text' with tensor shape (b, E).
        """
        return self.encoder.get_embeddings(text=query, modality='text')
    
    def forward(self, map_features, query):
        """
        Args:
            map_features: Tensor of shape (b, w, h, E) from your multi-head attention (each spatial location is an embedding).
            query: A text query.
        
        Returns:
            If method == "global": 
                Tensor of shape (b, 2) with the predicted (x,y) coordinates.
            If method == "hybrid": 
                dict with keys:
                    'regression' - coordinate prediction from global pooling branch (b, 2)
                    'heatmap'    - coordinate prediction from soft-argmax over a predicted heatmap (b, 2)
                    'heatmap_raw'- the refined (pre-softmax) heatmap (b, w, h)
        """
        # We always return a dictionary for consistency
        output = {}
        b, w, h, E = map_features.shape
        
        # Get the query embedding: assume encoder returns a dict with key 'text' (shape: (b, E))
        query_features = self.encode_query(query)['text']
        assert query_features.shape == (b, E), "Query embedding must have shape (b, E)"
        
        # ----- Global Regression Branch (Strategy 1) -----
        # Permute map_features to (b, E, w, h) for the pooling operation.
        map_features_perm = map_features.permute(0, 3, 1, 2)  # (b, E, w, h)
        pooled_feat = self.global_pool(map_features_perm).view(b, E)  # (b, E)
        # Concatenate pooled feature with query embedding.
        global_input = torch.cat([pooled_feat, query_features], dim=1)  # (b, 2E)
        regression_coords = self.regressor(global_input)  # (b, 2)
        output['reg_coords'] = regression_coords
        
        if self.method == "global":
            return output
        
        elif self.method == "hybrid":
            # ----- Heatmap Branch (Hybrid Multi-Task Learning) -----
            # First, compute a cosine similarity map between each spatial vector and the query embedding.
            # map_features: (b, w, h, E) and query_features: (b, E) => reshape query for broadcasting.
            x = map_features  # (b, w, h, E)
            y = query_features.view(b, 1, 1, E)  # (b, 1, 1, E)
            sim_map = self.cosine_similarity(x, y)  # (b, w, h)
            
            # Refine the similarity map via a small conv head.
            sim_map_unsq = sim_map.unsqueeze(1)  # (b, 1, w, h)
            refined_map = self.heatmap_conv(sim_map_unsq).squeeze(1)  # (b, w, h)
            
            # Apply softmax over the spatial locations (flatten w x h).
            refined_flat = refined_map.view(b, -1)
            heatmap_prob = F.softmax(refined_flat, dim=1).view(b, w, h)  # (b, w, h)
            
            # Create coordinate grids.
            device = heatmap_prob.device
            grid_x = torch.linspace(0, w - 1, steps=w, device=device)
            grid_y = torch.linspace(0, h - 1, steps=h, device=device)
            # Meshgrid with indexing such that grid_x corresponds to the w-dimension and grid_y to the h-dimension.
            grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')  # each is (w, h)
            # Expand to (1, w, h) for broadcasting.
            grid_x = grid_x.unsqueeze(0)
            grid_y = grid_y.unsqueeze(0)
            
            # Compute the soft-argmax (i.e. expectation) of the coordinates.
            pred_x = (heatmap_prob * grid_x).view(b, -1).sum(dim=1, keepdim=True)
            pred_y = (heatmap_prob * grid_y).view(b, -1).sum(dim=1, keepdim=True)
            heatmap_coords = torch.cat([pred_x, pred_y], dim=1)  # (b, 2)
            
            # (Optionally, you could scale the predicted coordinates by pixels_per_meter here.)
            output['softmax_coords'] = heatmap_coords # From soft-argmax over the heatmap
            output['value_map'] = refined_map # The raw (refined) heatmap (for visualization or auxiliary losses)
            return output  

