"""YOLOv8 vehicle detector with ByteTrack tracking."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# Shipped in the repo: hosted servers (e.g. Streamlit Community Cloud) can fail to download it
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"
# INT8-quantized ONNX export for CPU-only servers (tools/quantize_models.py); input size fixed at 416
CPU_WEIGHTS = DEFAULT_WEIGHTS.with_name("yolov8n_int8.onnx")
CPU_IMGSZ = 416

# COCO class ids used by the pretrained YOLOv8 weights
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    track_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    cls_name: str
    det_conf: float


class VehicleDetector:
    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        # Pick explicitly: no GPU on e.g. Streamlit Community Cloud
        self.device = device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
        if weights is None:
            weights = CPU_WEIGHTS if self.device == "cpu" and CPU_WEIGHTS.exists() else DEFAULT_WEIGHTS
        # Fall back to Ultralytics' auto-download if the file isn't there
        self.weights = str(weights) if Path(weights).exists() else Path(weights).name
        self.is_onnx = self.weights.endswith(".onnx")
        self.backend = "ONNX Runtime INT8" if self.is_onnx else ("PyTorch FP16" if self.device != "cpu" else "PyTorch FP32")
        self.model = YOLO(self.weights, task="detect")
        self._threads_set = False

    def reset(self):
        """Start a fresh tracker (new video or webcam session)."""
        self.model = YOLO(self.weights, task="detect")
        self._threads_set = False

    def _limit_onnx_threads(self):
        """Ultralytics opens ONNX Runtime with one thread per reported core; on a shared cloud host that
        is far more than we get (see cpu.py). Swap in a session with PyTorch's (capped) thread count."""
        self._threads_set = True
        try:
            import onnxruntime as ort
            backend = self.model.predictor.model.backend
            so = ort.SessionOptions()
            so.intra_op_num_threads = torch.get_num_threads()
            backend.session = ort.InferenceSession(self.weights, so, providers=["CPUExecutionProvider"])
        except Exception:  # Ultralytics internals changed: keep its own session
            pass

    def track(self, frame_bgr: np.ndarray, conf: float = 0.35, imgsz: int = 640) -> list[Detection]:
        results = self.model.track(
            frame_bgr,
            persist=True,
            conf=conf,
            imgsz=CPU_IMGSZ if self.is_onnx else imgsz,  # the ONNX model has a fixed input size
            classes=list(VEHICLE_CLASSES),
            tracker="bytetrack.yaml",
            device=self.device,
            quantize=16 if self.device != "cpu" and not self.is_onnx else None,  # fp16 on GPU only
            verbose=False,
        )[0]
        if self.is_onnx and not self._threads_set:
            self._limit_onnx_threads()

        boxes = results.boxes
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        return [
            Detection(int(i), tuple(b), VEHICLE_CLASSES.get(int(c), "vehicle"), float(p))
            for b, i, c, p in zip(xyxy, ids, clss, confs)
        ]
