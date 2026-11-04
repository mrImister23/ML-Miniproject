"""YOLOv8 vehicle detector with ByteTrack tracking."""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# COCO class ids used by the pretrained YOLOv8 weights
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    track_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    cls_name: str
    det_conf: float


class VehicleDetector:
    def __init__(self, weights: str = "yolov8n.pt", device: str | None = None):
        self.weights = weights
        self.device = device
        self.model = YOLO(weights)

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
            half=self.device != "cpu",
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
