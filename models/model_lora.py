import torch.nn as nn
import warnings
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F

# Custom imports
from utils.utils import soft_argmax_coords

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)


class TrainedLoraModel(nn.Module):
    def __init__(self, scene_encoder, query_encoder, top_k=1, neighborhood=0, nms_radius=2, device="cuda"):

        super().__init__()
        print("Initializing Lora only Model...")

        # Initialize Encoders
        self.scene_encoder = scene_encoder
        self.query_encoder = query_encoder

        # Set Hyperparameters for the model
        self.top_k = top_k
        self.neighborhood = neighborhood
        self.nms_radius = nms_radius
        self.device = device
        self.tau = 0.1  # Temperature for soft-argmax
        
        print("Trained Lora Model (Encoder finetuting) initialized.")
   
    def encode_descriptions(self, descriptions):
        """
        descriptions: List[List[str]] of shape (batch_size, variable_lengths)

        Returns:
            Tensor of shape (batch_size, max_seq_len, embedding_dim)
        """
        # 1) Encode each list of strings into a tensor of shape (seq_len_i, E)
        embedding_tensors = []
        for desc_list in descriptions:
            emb_dict = self.scene_encoder.get_embeddings(text=desc_list, modality='text')
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
    
    def encode_query(self, query):
        """
        Encodes a given query into an embedding.
        
        Args:
            query (str): The query to encode.
        
        Returns:
            dict: Contains a key 'text' with tensor shape (b, E) representing the query embedding.
        """
        return self.query_encoder.get_embeddings(text=query, modality='text')
    
    def cosine_similarity(self, x, y):
        """
        Computes the cosine similarity between two tensors.
        
        Args:
            x (Tensor): First tensor.
            y (Tensor): Second tensor.
        
        Returns:
            Tensor: Cosine similarity between x and y.
        """
        return F.cosine_similarity(x, y, dim=-1) 
    
    def forward(self, description, map_tensor, query, gt_coords):
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
        output = {}
        
        # 0) Normalize feature map
        feature_map = map_tensor.to(self.device)  # (b, H, W, E)
        feature_map = F.normalize(feature_map, p=2, dim=-1)
        b, H, W, E = feature_map.shape
        feat_flat = feature_map.view(b, H * W, E)
        
        # 1) encode descriptions → desc_embeds (b, k, E)
        description = self.encode_descriptions(description).to(self.device)
        description = F.normalize(description, p=2, dim=-1)
        _, K, _ = description.shape
        
        # 2) encode query → q_emb (b, E)
        query = self.encode_query(query)['text'].to(self.device)
        query = F.normalize(query, p=2, dim=-1)
        # Prepare query tensor
        query_tensor = query.unsqueeze(1).unsqueeze(2).expand(-1, H, W, -1)
        
        # Prepare descriptions for batch matrix multiplication
        desc_t = description.transpose(1, 2)
        
        # Compute cosine similarity per spatial location and description
        desc_value_flat = torch.bmm(feat_flat, desc_t)
        desc_value_map = desc_value_flat.view(b, H, W, K)
        desc_flat = desc_value_map.view(b, H * W, K).permute(0, 2, 1)  # → (b, K, H*W)
        
        # Initialize empty boolean mask
        mask = torch.zeros((b, H, W), dtype=torch.bool, device=feature_map.device)

        for i in range(b):  # batch
            for j in range(K):  # description
                # Get flattened scores and their indices
                scores = desc_flat[i, j]  # (H*W)
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                
                selected = []  # List of accepted (row, col)
                count = 0
                for idx in sorted_indices:
                    row, col = divmod(idx.item(), W)

                    # Check if it's far enough from already selected points
                    too_close = any((abs(row - r) <= self.nms_radius and abs(col - c) <= self.nms_radius) for (r, c) in selected)
                    if too_close:
                        continue
                    
                    # Add to selected and increment counter
                    selected.append((row, col))
                    count += 1
                    if count >= self.top_k:
                        break
                
                # Apply neighborhood for each selected coordinate
                for row, col in selected:
                    for dx in range(-self.neighborhood, self.neighborhood + 1):
                        for dy in range(-self.neighborhood, self.neighborhood + 1):
                            r2 = min(max(row + dx, 0), H - 1)
                            c2 = min(max(col + dy, 0), W - 1)
                            mask[i, r2, c2] = True

        # Reshape and apply mask
        mask = mask.unsqueeze(-1)  # (b, H, W, 1)
        feature_map = feature_map * mask

        # Compute similarity
        value_map = self.cosine_similarity(feature_map, query_tensor)

        # Add a trailing singleton channel to match (b, h, w, 1)
        value_map = value_map.view(b, H, W, 1)
        output["value_map"] = value_map

        # 2) Soft-argmax → (b, 2)
        coords = soft_argmax_coords(value_map, self.tau)
        output["coords"] = coords
        
        return output
    
    def update_tau(self, epoch):
        """
        Updates the tau value based on the current epoch.
        
        Args:
            epoch (int): The current epoch number.
        """
        if epoch >= self.tau_step:
            self.tau = self.tau_min
        else:
            self.tau = self.tau_max
    