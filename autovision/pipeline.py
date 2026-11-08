"""Full per-frame pipeline: detect -> track -> classify -> estimate distance -> draw HUD."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import hud
from .classifier import CarClassifier
from .detector import VehicleDetector
from .distance import estimate_distance, proximity_level
from .specs import CarSpec, lookup
from .tracker_state import TrackState

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "car_classifier.pt"
DEFAULT_NAMES = ROOT / "models" / "class_names.json"

# Classify cars and trucks (pickups/vans are in Stanford Cars); buses only get detected
CLASSIFIABLE = {"car", "truck"}


@dataclass
class PipelineConfig:
    det_conf: float = 0.35
    # From test-set calibration: model conf 0.25-0.3 is still ~74% correct
    cls_conf: float = 0.25       # below this the label is shown as a best guess with "?"
    classify_every: int = 5      # re-classify each track every N frames
    min_crop_px: int = 48        # smaller boxes are too far away to classify
    hfov_deg: float = 70.0       # camera horizontal field of view
    show_distance: bool = True
    show_specs: bool = True
    # None = pick for the hardware: full quality on a GPU; on a CPU-only server (e.g. Streamlit
    # Community Cloud) a smaller YOLO input and detecting every other frame, reusing the boxes in
    # between, which roughly doubles the frame rate at the cost of boxes lagging by one frame
    imgsz: int | None = None
    detect_every: int | None = None


@dataclass
class VehicleResult:
    track_id: int
    box: tuple[int, int, int, int]
    cls_name: str
    spec: CarSpec | None
    confidence: float
    distance_m: float
    proximity: str
    uncertain: bool = False   # spec is a best guess below cls_conf

    @property
    def display_name(self) -> str:
        if self.spec:
            return self.spec.short + ("?" if self.uncertain else "")
        return self.cls_name.title()  # too small/far to classify yet, or a bus


class AutoVisionPipeline:
    def __init__(self, classifier: CarClassifier | None = None, detector: VehicleDetector | None = None,
                 config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.detector = detector or VehicleDetector()
        on_cpu = self.detector.device == "cpu"
        if self.config.imgsz is None:
            self.config.imgsz = 416 if on_cpu else 640
        if self.config.detect_every is None:
            self.config.detect_every = 2 if on_cpu else 1
        self._detections = []
        self.classifier = classifier
        self.state = TrackState()
        self.frame_idx = 0
        self._last_t = None
        self.fps = 0.0

    @classmethod
    def load_classifier(cls, weights=DEFAULT_WEIGHTS, names=DEFAULT_NAMES) -> CarClassifier | None:
        if not Path(weights).exists() or not Path(names).exists():
            return None
        return CarClassifier(weights, names)

    def reset(self):
        self.detector.reset()
        self.state.reset()
        self.frame_idx = 0
        self._last_t = None
        self._detections = []

    def _crop(self, frame, box, pad=0.08):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        return frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list[VehicleResult]]:
        cfg = self.config
        self.frame_idx += 1
        h, w = frame_bgr.shape[:2]

        if cfg.detect_every <= 1 or self.frame_idx % cfg.detect_every == 1:
            self._detections = self.detector.track(frame_bgr, conf=cfg.det_conf, imgsz=cfg.imgsz)
        detections = self._detections

        # Batch-classify only the tracks that are due, to keep FPS high
        if self.classifier is not None:
            due = [
                d for d in detections
                if d.cls_name in CLASSIFIABLE
                and min(d.box[2] - d.box[0], d.box[3] - d.box[1]) >= cfg.min_crop_px
                and self.state.needs_classification(d.track_id, self.frame_idx, cfg.classify_every)
            ]
            if due:
                probs = self.classifier.predict_proba([self._crop(frame_bgr, d.box) for d in due])
                for d, p in zip(due, probs):
                    self.state.add(d.track_id, self.frame_idx, p)

        results = []
        for d in detections:
            mem = self.state.get(d.track_id, self.frame_idx)
            spec, conf, uncertain = None, 0.0, False
            proba = mem.mean_proba()
            if proba is not None:
                idx = int(proba.argmax())
                conf = float(proba[idx])
                # Unsure predictions are still shown, flagged as a best guess (on the test set the
                # plain top-1 beats a brand-first pick here: 55% vs 52% exact, 69% vs 67% brand)
                uncertain = conf < cfg.cls_conf
                spec = lookup(self.classifier.class_names[idx])
            dist = estimate_distance(d.box, d.cls_name, w, cfg.hfov_deg)
            results.append(VehicleResult(d.track_id, d.box, d.cls_name, spec, conf, dist, proximity_level(dist),
                                         uncertain))
        self.state.cleanup(self.frame_idx)

        now = time.perf_counter()
        if self._last_t is not None:
            inst = 1.0 / max(now - self._last_t, 1e-6)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last_t = now

        out = frame_bgr.copy()
        for v in sorted(results, key=lambda r: -r.distance_m):  # draw closest last (on top)
            hud.draw_vehicle(out, v, cfg.show_distance)
        recognised = [v for v in results if v.spec]
        if cfg.show_specs and recognised:
            hud.draw_spec_panel(out, min(recognised, key=lambda r: r.distance_m))
        hud.draw_fps(out, self.fps, len(results))
        return out, results


if __name__ == "__main__":
    import sys

    import cv2

    src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "samples" / "street.jpg")
    pipe = AutoVisionPipeline(classifier=AutoVisionPipeline.load_classifier())
    img = cv2.imread(src)
    if img is None:
        sys.exit(f"Cannot read {src}")
    annotated, vehicles = pipe.process_frame(img)
    out_path = Path(src).with_name(Path(src).stem + "_annotated.jpg")
    cv2.imwrite(str(out_path), annotated)
    for v in vehicles:
        print(f"#{v.track_id} {v.display_name} conf={v.confidence:.2f} dist={v.distance_m:.1f}m")
    print("Saved", out_path)
