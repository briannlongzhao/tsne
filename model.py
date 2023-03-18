import os.path
from typing import Iterator, OrderedDict
import numpy as np
import timm
import sys
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str, batch_size: int):
        super().__init__()
        assert model_name in timm.list_models(pretrained=True)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299,299)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img):
        # (N, C=3, H, W) -> (N, d)
        embs = self.model(img)
        return embs.cpu().numpy()
    
    @torch.no_grad()
    def compute_embedding(self, images: Iterator[np.ndarray]) -> np.ndarray:
        segs = [
            self.transform(Image.fromarray(image))
            for image in images
            if image.shape[-1] == 3
        ]
        # (N, C=3, H, W)
        segs = torch.stack(segs, dim=0)
        embs = []
        for seg in segs.split(self.batch_size):
            # each seg (bsz, 3, H, W)
            emb = self(seg.to(self.device))
            embs.append(emb)
        # (N, d)
        return np.concatenate(embs, axis=0)
