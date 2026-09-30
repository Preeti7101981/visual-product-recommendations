"""
Feature Creation
================
Defines the image transform chain and the embedding model used throughout
this project: feature extraction, embedding generation, recommendation,
evaluation, and deployment all use this same module and the same model
(ResNet50). There is no alternate/lite model anywhere in this project.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmbeddingModel(nn.Module):
    """ResNet50 backbone with the classification head removed.

    Used purely as a fixed feature extractor (unsupervised use): class
    predictions are never used, only the 2048-dimensional penultimate-layer
    activations, which serve as a general-purpose visual embedding.
    """

    def __init__(self):
        super().__init__()
        assert config.MODEL_NAME == "resnet50", (
            f"config.MODEL_NAME is '{config.MODEL_NAME}' but feature_creation.py "
            "only implements resnet50. Keep these in sync."
        )
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.backbone.eval()
        self.backbone.to(DEVICE)

    @torch.no_grad()
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image_tensor.to(DEVICE))
        return features.flatten(start_dim=1)


def load_and_transform_image(image_path: str) -> torch.Tensor:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        tensor = IMAGE_TRANSFORM(img)
    return tensor.unsqueeze(0)


def get_embedding(model: EmbeddingModel, image_path: str) -> torch.Tensor:
    tensor = load_and_transform_image(image_path)
    embedding = model(tensor).squeeze(0)
    embedding = embedding / embedding.norm(p=2)  # L2-normalize
    return embedding.cpu()
