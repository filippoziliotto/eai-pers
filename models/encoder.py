# Base imports
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union, List
import types
import sys

# Add the path to the lavis directory
sys.path.append('LAVIS/')

# Lavis imports
from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures

# Lora imports
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

class Blip2Encoder:
    encoder: torch.nn.Module
    vis_processors: torch.nn.Module
    text_processors: torch.nn.Module
    
    def __init__(self, device: str = 'cpu', freeze_encoder: bool = True, use_lora: bool = True):
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.use_lora = use_lora

    def initialize(self, name: str = "blip2_image_text_matching", model_type: str = "pretrain") -> torch.nn.Module:
        """
        Initialize the BLIP2ITM model.
        
        Args:
            name (str): The name of the BLIP2ITM model.
            model_type (str): The type of the BLIP2ITM model.
        """
        
        self.encoder, self.vis_processors, self.text_processors = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=not self.use_lora,
            device=self.device,
        )
        
        # Freeze the model parameters
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Inject LoRA nella Q-Former
        if self.use_lora:
            print("Injecting LoRA into BLIP-2 QFormer...")
            # Prendi l'oggetto QFormer.bert
            orig_bert = self.encoder.Qformer.bert
            
            # Se è già un PeftModel, estrai il modello base; altrimenti usa orig_bert direttamente
            base_bert = orig_bert.base_model if isinstance(orig_bert, PeftModel) else orig_bert
            
            # Configurazione LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            # Inietta LoRA sul modello base
            lora_bert = get_peft_model(base_bert, lora_config)
            lora_bert.print_trainable_parameters()
            
            # Riassegna il modello modificato al wrapper QFormer
            self.encoder.Qformer.bert = lora_bert
            
        # Patch the extract_features method to support PEFT models in "text" mode
        self._monkeypatch_extract_features(self.encoder)
            
        return self.encoder
    
    def _monkeypatch_extract_features(self, encoder):
        """
        Replace the original extract_features function with a patched version
        that correctly handles PEFT-wrapped QFormer BERT in 'text' mode.
        """
        original_method = encoder.extract_features

        def patched_extract_features(self_, samples, mode="multimodal"):
            if mode != "text":
                return original_method(samples, mode)

            # Handle text mode with PEFT-wrapped QFormer
            image = samples.get("image")
            caption = samples.get("text_input")
            assert caption is not None, "Text input is required for mode 'text'"

            text = self_.tokenizer(caption, return_tensors="pt", padding=True).to(self_.device)

            # Retrieve the correct BERT model depending on whether LoRA is used
            bert_model = (
                self_.Qformer.bert.base_model
                if isinstance(self_.Qformer.bert, PeftModel)
                else self_.Qformer.bert
            )

            text_output = bert_model(
                input_ids=text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )

            text_embeds = text_output.last_hidden_state
            text_features = self_.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

            return BlipOutputFeatures(
                image_embeds=None,
                image_embeds_proj=None,
                text_embeds=text_embeds,
                text_embeds_proj=text_features,
                multimodal_embeds=None,
            )

        # Bind the method to the encoder instance
        encoder.extract_features = types.MethodType(patched_extract_features, encoder)

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
    
    def preprocess_inputs(self, image: np.ndarray, text: Union[str, List[str]]) -> Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
            """
            Preprocess the input image and text.

            Args:
                image (numpy.ndarray): The input image as a numpy array.
                text (Union[str, List[str]]): The input text or list of texts.

            Returns:
                Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]: The preprocessed image and text.
            """
            if isinstance(image, torch.Tensor):
                image = Image.fromarray(image.cpu().numpy())
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pass
            
            img, txt = None, [None]

            if image is not None:
                img = self.vis_processors['eval'](image).unsqueeze(0).to(self.device)
                
            # Handling batched data text input
            txt = [self.text_processors['eval'](t) for t in text]

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
        
        # Handle batched data text inputs
        sample = {"image": img, "text_input": txt}

        if modality in ["image"]:
            assert image is not None
            #with torch.inference_mode():
            image_embedding = self.encoder.extract_features(sample, mode="image").image_embeds[:,0,:]
        elif modality in ["text"]:
            assert text is not None
            #with torch.inference_mode():
            text_embeddings = [self.encoder.extract_features({"image": img, "text_input": t}, mode="text").text_embeds[:,0,:] for t in txt]
            text_embedding = torch.stack(text_embeddings).squeeze(1)
        elif modality in ['both']:
            assert image is not None and text is not None
            #with torch.inference_mode():
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