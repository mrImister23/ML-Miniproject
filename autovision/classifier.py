"""MobileNetV3 fine-grained make & model classifier (Stanford Cars, 196 classes)."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_large

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(num_classes: int) -> nn.Module:
    """MobileNetV3-Large with a new classification head. Shared with the training notebook."""
    model = mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


class CarClassifier:
    def __init__(self, weights_path: str | Path, names_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names: list[str] = json.loads(Path(names_path).read_text(encoding="utf-8"))
        self.model = build_model(len(self.class_names))
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.half = self.device == "cuda"
        if self.half:
            self.model.half()

    @staticmethod
    def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return x.transpose(2, 0, 1)

    @torch.inference_mode()
    def predict_proba(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Return softmax probabilities, shape (len(crops), num_classes)."""
        if not crops_bgr:
            return np.empty((0, len(self.class_names)), dtype=np.float32)
        batch = torch.from_numpy(np.stack([self._preprocess(c) for c in crops_bgr])).to(self.device)
        if self.half:
            batch = batch.half()
        logits = self.model(batch).float()
        return torch.softmax(logits, dim=1).cpu().numpy()
