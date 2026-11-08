"""YOLOv8 vehicle detector with ByteTrack tracking."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# Shipped in the repo: hosted servers (e.g. Streamlit Community Cloud) can fail to download it
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"

# COCO class ids used by the pretrained YOLOv8 weights
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    track_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    cls_name: str
    det_conf: float


class VehicleDetector:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str | None = None):
        # Fall back to Ultralytics' auto-download if the file isn't there
        self.weights = str(weights) if Path(weights).exists() else Path(weights).name
        # Pick explicitly: no GPU on e.g. Streamlit Community Cloud
        self.device = device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
        self.model = YOLO(self.weights)

    def reset(self):
        """Start a fresh tracker (new video or webcam session)."""
        self.model = YOLO(self.weights)

    def track(self, frame_bgr: np.ndarray, conf: float = 0.35, imgsz: int = 640) -> list[Detection]:
        results = self.model.track(
            frame_bgr,
            persist=True,
            conf=conf,
            imgsz=imgsz,
            classes=list(VEHICLE_CLASSES),
            tracker="bytetrack.yaml",
            device=self.device,
            quantize=16 if self.device != "cpu" else None,  # fp16 on GPU only
            verbose=False,
        )[0]

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
