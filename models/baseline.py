import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)


class BaselineModel(nn.Module):
    def __init__(self, encoder, type, device):
        """
        Initializes the RetrievalMapModel.

        Args:
            encoder (Blip2Encoder): The encoder model.
            type (str): The type of baseline to use. Options are "random", "center", or None.
            device (str): The device for model computation.
        """
        super().__init__()
        print("Initializing Baseline...")
        
        # Which baseline to use
        self.baseline_type = type
        if self.baseline_type:
            print(f"Using {self.baseline_type} baseline.")
            
        self.encoder = encoder
        self.device = device

        print("Baseline initialized.")

    def forward(self, feature_map, query):
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
        b, w, h, E = feature_map.shape
        
        if self.baseline_type in ["random"]:
            # For random baseline, just return a random index
            output["value_map"] = torch.rand((b, w, h, 1))
            output["max_value"] = torch.rand((b, 1))
            output["max_index"] = torch.randint(0, w * h, (b,)).view(b, w, h)
            return output
            
        elif self.baseline_type in ["center"]:
            # For center baseline, just return the center of the feature map
            output["value_map"] = torch.rand((b, w, h, 1))
            output["max_value"] = torch.rand((b, 1))
            output["max_index"] = torch.tensor([w // 2, h // 2]).expand(b, -1)
            return output
        
        elif self.baseline_type in ["vlfm"]:
            # Encode the query
            query_tensor = self.encode_query(query)
            assert query_tensor.shape == (b, E), "Query embedding must have shape (b, E)"
            
            # Calculate cosine similarity
            value_map = self.cosine_similarity(
                feature_map, 
                query_tensor
            ) # b x w x h
            output["value_map"] = value_map.view(b, w, h, 1)
            
            # Get max element in the value map for each batch
            max_value_map, max_index = value_map.view(b, -1).max(dim=-1)
            output["max_value"] = max_value_map
            output["max_index"] = max_index.view(b, w, h)  
            return output
        
        else:
            raise NotImplementedError(f"Baseline type '{self.baseline_type}' is not implemented.")
    
    def encode_query(self, query):
        """
        Encodes a given query into an embedding.
        
        Args:
            query (str): The query to encode.
        
        Returns:
            dict: Contains a key 'text' with tensor shape (b, E) representing the query embedding.
        """
        return self.encoder.get_embeddings(text=query, modality='text')
    
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
    