import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings
import os
from openai import OpenAI

# Local imports
from models.stages.zs_stage import ZeroShotCosineModel

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
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        print("Initializing Baseline...")
        
        self.encoder = encoder
        self.device = device
        
        self.keyword_to_mask = {} 

        
        # Which baseline to use
        self.baseline_type = type
        if self.baseline_type:
            print(f"Using {self.baseline_type} baseline.")
            
        if self.baseline_type in ["zs_cosine"]:
            # Initialize the zero-shot cosine model
            self.zs_model = ZeroShotCosineModel(encoder=encoder).to(device)

        print("Baseline initialized.")


    def forward(self, map_tensor, description, query, gt_coords=None):
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
            feature_map = map_tensor
            b, w, h, E = feature_map.shape
            
            if self.baseline_type in ["random"]:
                # For random baseline, just return a random index
                output["value_map"] = torch.rand((b, w, h, 1))
                output["max_value"] = torch.rand((b, 1))

                x = torch.randint(0, w, (b,), device=self.device)
                y = torch.randint(0, h, (b,), device=self.device)

                # output["coords"] = torch.randint(0, w * h, (b,)).view(b, w, h)
                output["coords"] = torch.stack([x, y], dim=1)  # [b, 2]
                
                return output
        
            
            elif self.baseline_type in ["center"]:
                # For center baseline, just return the center of the feature map
                output["value_map"] = torch.rand((b, h, w, 1))
                output["max_value"] = torch.rand((b, 1))
                output["coords"] = torch.tensor([h// 2, w // 2]).expand(b, -1).to(self.device)
                return output
            
            elif self.baseline_type in ["vlfm"]:
                # Encode the query
                #query_tensor = self.encode_query(query)
                query_tensor = self.encode_query(query)["text"]
                print("query_tensor shape:", query_tensor.shape)

                assert query_tensor.shape == (b, E), "Query embedding must have shape (b, E)"

                query_tensor = query_tensor.unsqueeze(1).unsqueeze(1)

                # Calculate cosine similarity
                value_map = self.cosine_similarity(
                    feature_map, 
                    query_tensor
                ) # b x w x h
                output["value_map"] = value_map.view(b, w, h, 1)
                
                # Get max element in the value map for each batch and convert to coordinates
                max_value_map, max_index = value_map.view(b, -1).max(dim=-1)
                output["max_value"] = max_value_map
                # output["coords"] = max_index.view(b, w, h)
                output["coords"] = torch.stack([
                    max_index // h,
                    max_index % h
                ], dim=1)  # shape: (b, 2)++
                return output
            
            elif self.baseline_type in ["zs_cosine"]:
                # Encode query embeddings
                query_tensor = self.encode_query(query)['text']
                assert query_tensor.shape == (b, E), "Query embedding must have shape (b, E)"
                
                # Encode description embeddings
                description_tensor = self.zs_model.encode_descriptions(description)
                assert description_tensor.shape[0] == b, "Description embedding must have shape (b, k, E)"
                
                # Get the max value and index from the zero-shot cosine model
                max_index, max_val = self.zs_model.forward(
                    feature_map=feature_map,
                    query_tensor=query_tensor,
                    description_tensor=description_tensor,
                    top_k=4,
                    neighborhood=0,
                    nms_radius=2,
                )

                output["max_value"] = max_val
                output["coords"] = torch.stack([max_index // w, max_index % w], dim=-1) 
                return output
            
            elif self.baseline_type in ["llm_parse_similarity"]:
                # Use the parser to extract the key object from the description
                parsed_queries = self.llm_parse(description)  # List[str]

                # Embed the parsed query
                query_tensor = self.encode_query(parsed_queries)["text"]  # (b, E)
                print("Expected E =", E) 
                print("b =", b)
                print("query_tensor.shape =", query_tensor.shape)
                assert query_tensor.shape == (b, E), f"Query embedding must have shape (b, E), got {query_tensor.shape}"

                # Expand dimensions for broadcasting with feature_map (b, w, h, E)
                query_tensor = query_tensor.unsqueeze(1).unsqueeze(1)  # (b, 1, 1, E)

                # Cosine similarity
                value_map = self.cosine_similarity(feature_map, query_tensor)  # (b, w, h)
                output["value_map"] = value_map.view(b, w, h, 1)

                # Get the maximum value and its index in the value map
                max_value_map, max_index = value_map.view(b, -1).max(dim=-1)
                output["max_value"] = max_value_map
                output["coords"] = torch.stack([
                    max_index // h,
                    max_index % h
                ], dim=1)  # shape: (b, 2)

                return output

            elif self.baseline_type in ["llm_parse_masked_similarity"]:
                # 1) Keywords dal LLM (una per esempio -> len == b)
                keywords = self.llm_parse(description)  # List[str] len == b

                # 2) Embedding delle keyword
                query_tensor = self.encode_query(keywords)["text"]  # atteso (b, E)
                if query_tensor.shape[0] != b:
                    # sicurezza: se per qualche motivo arriva 1 x E, fai broadcast
                    query_tensor = query_tensor[:1].expand(b, -1)

                # 3) Similarity map (cosine a ogni cella)
                # feature_map: (b, w, h, E) —> portiamo query a (b,1,1,E)
                query_tensor = query_tensor.unsqueeze(1).unsqueeze(1)  # (b,1,1,E)
                value_map = self.cosine_similarity(feature_map, query_tensor)  # (b, w, h)

                # 4) Maschera per batch
                mask = self.get_mask_for_batch(keywords, w, h, b, self.device)  # (b,w,h)

                # 5) Applica mask + argmax
                max_value_map, max_index = self.apply_mask_and_argmax(value_map, mask)

                output["value_map"] = value_map.view(b, w, h, 1)
                output["max_value"] = max_value_map
                output["coords"] = torch.stack([
                    max_index // h,
                    max_index % h
                ], dim=1)  # (b,2)
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

    def llm_parse(self, descriptions):
        """
        Uses GPT to extract the main object from a list of descriptions.

        Args:
            descriptions (list[str]): List of textual descriptions.

        Returns:
            list[str]: List of extracted key words/phrases for each description.
        """
        parsed = []
        for desc in descriptions:
        # Combine all sentences in a single string
            if isinstance(desc, list):
                combined_desc = " ".join(sentence.strip() for sentence in desc)
            else:
                combined_desc = desc.strip()

            prompt = (
                "Extract the main object to locate from the following description:\n"
                f"\"{combined_desc}\"\n"
                "Reply with one or two keywords only, without any other explanation."
            )

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
                n=1,
            )

            key_object = response.choices[0].message.content.strip().lower()
            parsed.append(key_object)
        return parsed

    def get_object_mask(self, keyword, w, h, device):
        """
        Ritorna la maschera (w,h) per una singola keyword.
        Usa self.keyword_to_mask se disponibile, altrimenti ritorna una maschera piena.
        """
        mask = None
        if hasattr(self, "keyword_to_mask") and isinstance(self.keyword_to_mask, dict):
            if keyword in self.keyword_to_mask:
                mask = self.keyword_to_mask[keyword]
                if mask.dim() == 2 and mask.shape != (w, h):
                    mask = mask.float().unsqueeze(0).unsqueeze(0)
                    mask = F.interpolate(mask, size=(w, h), mode='nearest').squeeze(0).squeeze(0)
        if mask is None:
            mask = torch.ones((w, h), device=device)
        return mask.clamp(0, 1)


    def get_mask_for_batch(self, keywords, w, h, b, device):
        return torch.stack([self.get_object_mask(kw, w, h, device) for kw in keywords], dim=0)


    def apply_mask_and_argmax(self, value_map, mask):
        """
        value_map: (b, w, h) con cosine similarity
        mask:     (b, w, h) con 0/1 (o soft in [0,1])
        Ritorna: max_value (b,), max_index (b,)
        Applica la maschera: fuori dal mask -> very negative (per non vincere il max).
        """
        very_neg = torch.finfo(value_map.dtype).min / 2  # o -1e9
        masked = torch.where(mask > 0, value_map, very_neg)
        max_val, max_idx = masked.view(masked.shape[0], -1).max(dim=-1)
        return max_val, max_idx
