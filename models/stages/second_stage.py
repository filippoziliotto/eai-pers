import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinatePredictionModel(nn.Module):
    def __init__(self, encoder, method="global", hidden_dim=128):
        """
        A coordinate prediction model that can operate in two modes:
        
        1. 'global'  - Global Pooling + FC Regression.
        2. 'hybrid'  - Hybrid Multi-Task: regression branch plus heatmap prediction (via soft-argmax).
        
        The regression branch is designed to predict normalized coordinates in the [0, 1] range,
        which are then rescaled to pixel coordinates using the spatial dimensions of the input feature map.
        The heatmap branch directly computes expected pixel coordinates from a softmax over a refined similarity map.
        
        Args:
            encoder: An encoder with a method get_embeddings(text=..., modality='text') that returns
                     a dict with key 'text' and tensor shape (b, E). It is assumed that encoder.embedding_dim exists.
            method (str): Either "global" or "hybrid". Determines which branches are executed.
            hidden_dim (int): Hidden layer dimension for the regression MLP.
        """
        super(CoordinatePredictionModel, self).__init__()
        self.encoder = encoder
        self.method = method
        
        # We'll use cosine similarity (over the embedding dimension) for the heatmap branch.
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
        
        # --- Global Regression Branch (Strategy 1) ---
        # Expecting map_features of shape (b, w, h, E), we first permute to (b, E, w, h)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # The encoder is assumed to produce embeddings of size encoder.embedding_dim.
        feat_dim = encoder.encoder.itm_head.in_features
        # We concatenate the pooled map feature (shape: (b, feat_dim)) with the query embedding (b, feat_dim)
        # and then use an MLP to regress normalized (x,y) coordinates in [0, 1], which are then scaled by (w, h).
        self.regressor = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()  # Forces output to [0, 1]
        )
        
        if self.method == "hybrid":
            # --- Heatmap Branch (part of Strategy 5) ---
            # Computes a cosine similarity map between map_features and the query embedding.
            # A small convolutional head is used to refine this similarity map.
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
            dict: Contains a key 'text' with tensor shape (b, E) representing the query embedding.
        """
        return self.encoder.get_embeddings(text=query, modality='text')
    
    def forward(self, map_features, query):
        """
        Processes the input map features and text query to predict coordinates.
        
        Args:
            map_features: Tensor of shape (b, w, h, E) where each spatial location is an embedding.
            query: A text query.
        
        Returns:
            dict: If method == "global":
                    {
                        'regression_coords': Tensor of shape (b, 2) with pixel coordinates computed by scaling 
                                             normalized predictions (in [0,1]) with (w, h).
                    }
                  If method == "hybrid":
                    {
                        'regression_coords': Tensor of shape (b, 2) from the regression branch (pixel coordinates),
                        'heatmap_coords':    Tensor of shape (b, 2) from soft-argmax over the heatmap (pixel coordinates),
                        'value_map':       Tensor of shape (b, w, h) containing the refined similarity map before softmax.
                    }
                    
        Note:
            - The regression branch outputs normalized coordinates (via a sigmoid) which are then scaled by (w, h).
              If your ground truth coordinates are defined in the range [0, w-1] and [0, h-1], consider scaling by (w-1, h-1).
            - The heatmap branch computes expected coordinates directly in pixel space.
        """
        output = {}
        b, w, h, E = map_features.shape
        
        # Get the query embedding; expected to have shape (b, E)
        query_features = self.encode_query(query)['text']
        assert query_features.shape == (b, E), "Query embedding must have shape (b, E)"
        
        # ----- Global Regression Branch (Strategy 1) -----
        # Permute map_features to (b, E, w, h) for pooling.
        map_features_perm = map_features.permute(0, 3, 1, 2)  # (b, E, w, h)
        pooled_feat = self.global_pool(map_features_perm).view(b, E)  # (b, E)
        # Concatenate pooled feature with query embedding.
        global_input = torch.cat([pooled_feat, query_features], dim=1)  # (b, 2E)
        # Predict normalized coordinates in [0, 1]
        normalized_coords = self.regressor(global_input)  # (b, 2)
        # Scale normalized coordinates to pixel coordinates.
        pixel_coords = normalized_coords * torch.tensor([w, h], device=normalized_coords.device, dtype=normalized_coords.dtype)
        output['regression_coords'] = pixel_coords
        
        if self.method == "global":
            return output
        
        elif self.method == "hybrid":
            # ----- Heatmap Branch (Hybrid Multi-Task Learning) -----
            # Compute cosine similarity between each spatial vector and the query embedding.
            sim_map = self.cosine_similarity(map_features, query_features.view(b, 1, 1, E))  # (b, w, h)
            
            # Refine the similarity map with a convolutional head.
            refined_map = self.heatmap_conv(sim_map.unsqueeze(1)).squeeze(1)  # (b, w, h)
            
            # Apply softmax over the flattened spatial dimensions.
            heatmap_prob = F.softmax(refined_map.view(b, -1), dim=1).view(b, w, h)  # (b, w, h)
            
            # Create coordinate grids corresponding to pixel indices.
            device = heatmap_prob.device
            grid_x = torch.linspace(0, w - 1, steps=w, device=device)
            grid_y = torch.linspace(0, h - 1, steps=h, device=device)
            grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')  # each is (w, h)
            grid_x = grid_x.unsqueeze(0)  # (1, w, h)
            grid_y = grid_y.unsqueeze(0)  # (1, w, h)
            
            # Compute expected pixel coordinates (soft-argmax).
            pred_x = (heatmap_prob * grid_x).view(b, -1).sum(dim=1, keepdim=True)
            pred_y = (heatmap_prob * grid_y).view(b, -1).sum(dim=1, keepdim=True)
            heatmap_coords = torch.cat([pred_x, pred_y], dim=1)  # (b, 2)
            
            output['heatmap_coords'] = heatmap_coords
            output['value_map'] = refined_map  # For visualization or auxiliary losses
            return output
