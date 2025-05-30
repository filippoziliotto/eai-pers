# Importing necessary libraries
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class ZeroShotCosineModel(nn.Module):
    def __init__(self, encoder):
        """
        Initializes the MapAttentionModel with the given parameters.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            device (torch.device): The device to run the model on.
        """
        super(ZeroShotCosineModel, self).__init__()
        self.encoder = encoder
        
        # 1) Un‐normalized 3×3 Gaussian kernel (sum = 16)
        raw_gauss = torch.tensor([
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.]
        ])
        self.register_buffer("raw_gauss", raw_gauss)

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
    
    def cosine_similarity(self, x, y):
        """
        Computes the cosine similarity between two tensors.
        
        Args:
            x (Tensor): First tensor.
            y (Tensor): Second tensor.
        
        Returns:
            Tensor: Cosine similarity between x and y.
        """
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)    

    def forward(self, feature_map, query_tensor, description_tensor, top_k=1, neighborhood=0, nms_radius=2):
        # Normalize feature and description embeddings
        feat_norm = F.normalize(feature_map, p=2, dim=-1)
        desc_norm = F.normalize(description_tensor, p=2, dim=-1)
        
        # Get shapes and flatten spatial dimensions
        b, H, W, E = feat_norm.shape
        _, K, _ = desc_norm.shape
        feat_flat = feat_norm.view(b, H * W, E)
        
        # Prepare descriptions for batch matrix multiplication
        desc_t = desc_norm.transpose(1, 2)
        
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
                    too_close = any((abs(row - r) <= nms_radius and abs(col - c) <= nms_radius) for (r, c) in selected)
                    if too_close:
                        continue
                    
                    # Add to selected and increment counter
                    selected.append((row, col))
                    count += 1
                    if count >= top_k:
                        break
                
                # Apply neighborhood for each selected coordinate
                for row, col in selected:
                    for dx in range(-neighborhood, neighborhood + 1):
                        for dy in range(-neighborhood, neighborhood + 1):
                            r2 = min(max(row + dx, 0), H - 1)
                            c2 = min(max(col + dy, 0), W - 1)
                            mask[i, r2, c2] = True
        
        # Reshape and apply mask
        mask = mask.unsqueeze(-1)  # (b, H, W, 1)
        feature_map = feature_map * mask

        # Prepare query tensor
        query_tensor = query_tensor.unsqueeze(1).unsqueeze(2).expand(-1, H, W, -1)

        # Compute similarity
        value_map = self.cosine_similarity(feature_map, query_tensor)

        # Get max value and index
        max_value_map, max_index = value_map.view(b, -1).max(dim=-1)
        
        return max_index, max_value_map
    