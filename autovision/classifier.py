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
    def __init__(self, weights_path: str | Path, names_path: str | Path, device: str | None = None,
                 onnx_path: str | Path | None = None, threads: int | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
        self.class_names: list[str] = json.loads(Path(names_path).read_text(encoding="utf-8"))
        self.session = None
        if self.device == "cpu" and onnx_path and Path(onnx_path).exists():
            # ONNX model on CPU (see tools/quantize_models.py): much faster than PyTorch there
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            self.backend = "ONNX Runtime " + ("INT8" if "int8" in Path(onnx_path).name else "FP32")
            return
        self.backend = "PyTorch FP16" if self.device == "cuda" else "PyTorch FP32"
        self.model = build_model(len(self.class_names))
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.half = self.device == "cuda"
        if self.half:
            self.model.half()
        self.predict_proba([np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)])  # warm up before the first video frame

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
        if self.session is not None:
            logits = self.session.run(None, {"images": np.stack([self._preprocess(c) for c in crops_bgr])})[0]
            return torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        batch = torch.from_numpy(np.stack([self._preprocess(c) for c in crops_bgr])).to(self.device)
        if self.half:
            batch = batch.half()
        # cuDNN 9 on this GPU stalls for 3-8 s the first time it sees a new batch size or new kind of
        # input, which froze the video whenever the cars in view changed. PyTorch's own kernels never
        # stall and are only a few ms slower for a model this small.
        with torch.backends.cudnn.flags(enabled=False):
            logits = self.model(batch).float()
        return torch.softmax(logits, dim=1).cpu().numpy()
