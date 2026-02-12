import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPTextEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.language_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Freeze pretrained model parameters
        for p in self.language_model.parameters():
            p.requires_grad = False
        self.head = nn.Linear(512, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        feats = self.language_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        feats = self.head(feats)
        return feats
