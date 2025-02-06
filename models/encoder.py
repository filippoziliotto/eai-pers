# Base imports
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union

import sys
import os

# Add the path to the lavis directory
sys.path.append('LAVIS/')

# Lavis imports
from lavis.models import load_model_and_preprocess

class Blip2Encoder:
    encoder: torch.nn.Module
    vis_processors: torch.nn.Module
    text_processors: torch.nn.Module
    
    def __init__(self, device: str = 'cpu', freeze_encoder: bool = True):
        self.device = device
        self.freeze_encoder = freeze_encoder

    def initialize(self, name: str = "blip2_image_text_matching", model_type: str = "pretrain"):
        """
        Initialize the BLIP2ITM model.
        
        Args:
            name (str): The name of the BLIP2ITM model.
            model_type (str): The type of the BLIP2ITM model.
        """
        
        self.encoder, self.vis_processors, self.text_processors = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=self.device,
        )
        
        # Freeze the model parameters
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
        return self.encoder

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            
        pil_img = Image.fromarray(image)
        img = self.vis_processors(pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors(txt)
        with torch.inference_mode():
            cosine = self.encoder({"image": img, "text_input": txt}, match_head="itc").item()
        return cosine
    
    def preprocess_inputs(self, image: np.ndarray, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the input image and text.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            text (str): The input text.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The preprocessed image and text.
        """
        if isinstance(image, torch.Tensor):
            image = Image.fromarray(image.cpu().numpy())
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, PIL.Image.Image):
            pass

        img = self.vis_processors(image).unsqueeze(0).to(self.device)
        txt = self.text_processors(text)

        return img, txt
    
    def get_embeddings(
        self,
        image: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        modality: str = "both",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve embeddings for the specified modality.

        Args:
            image (numpy.ndarray, optional): The input image as a numpy array.
            text (str, optional): The input text.
            modality (str): The modality for which embeddings are retrieved. 
                            Options: "image", "text", "both".

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                The embeddings for the specified modality.
        """
        image_embedding, text_embedding, multimodal_embedding = None, None, None
        img, txt = self.preprocess_inputs(image, text)
        sample = {"image": img, "text_input": [txt]}

        if modality in ["image"]:
            assert image is not None
            with torch.inference_mode():
                image_embedding = self.encoder.extract_features(sample, mode="image").image_embeds[:,0,:]
        elif modality in ["text"]:
            assert text is not None
            with torch.inference_mode():
                text_embedding = self.encoder.extract_features(sample, mode="text").text_embeds[:,0,:]
        elif modality in ['both']:
            assert image is not None and text is not None
            with torch.inference_mode():
                image_embedding = self.encoder.extract_features(sample, mode="image").image_embeds[:,0,:]
                text_embedding = self.encoder.extract_features(sample, mode="text").text_embeds[:,0,:]
        elif modality in ['multimodal']:
                multimodal_embedding = self.encoder.extract_features(sample).multimodal_embeds[:,0,:]

        # Return dictionary of embeddings
        embeddings = {
            "image": image_embedding,
            "text": text_embedding,
            "multimodal": multimodal_embedding
        }
        
        # Normalize embeddings
        for key, value in embeddings.items():
            if value is not None:
                embeddings[key] = self.normalize_tensor(value)

        return embeddings

    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor along the last dimension.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        if tensor is None or []:
            return None
        
        return tensor / tensor.norm(dim=-1, keepdim=True)