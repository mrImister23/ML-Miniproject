"""Export both models to ONNX, quantize them to INT8 and compare FP32 / FP16 / INT8.

    python tools/quantize_models.py            # export + quantize + benchmark
    python tools/quantize_models.py --bench    # benchmark only (models already exported)

Writes models/yolov8n.onnx, models/yolov8n_int8.onnx, models/car_classifier.onnx,
models/car_classifier_int8.onnx and models/quantization_results.json.

CPU timings use 2 threads, like a Streamlit Community Cloud server. INT8 uses static
post-training quantization (ONNX Runtime, QDQ format, per-channel weights) calibrated on street
frames (detector) and Stanford Cars training images (classifier). Accuracy:
  - classifier: Top-1 / Top-5 on Stanford Cars test images, preprocessed exactly like the app
  - detector:   agreement with the FP32 model's vehicle boxes on the demo video
                (recall / precision at IoU 0.5, FP32 boxes taken as reference)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autovision.classifier import IMG_SIZE, CarClassifier, build_model  # noqa: E402
from autovision.detector import VEHICLE_CLASSES  # noqa: E402

MODELS = ROOT / "models"
VIDEO = ROOT / "samples" / "stanford_demo.mp4"
DET_IMGSZ = 416          # the app's YOLO input size on CPU
THREADS = 2              # like Streamlit Community Cloud
N_TEST = 2000            # Stanford Cars test images for the accuracy check
torch.set_num_threads(THREADS)
cv2.setNumThreads(THREADS)


# ------------------------------------------------------------------ data
def video_frames(step=4, limit=120):
    cap, frames, i = cv2.VideoCapture(str(VIDEO)), [], 0
    while len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    return frames


def letterbox(img, size=DET_IMGSZ):
    """Same as Ultralytics: keep aspect, pad with grey 114, BGR->RGB, 0-1, NCHW."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(out[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def stanford(split, n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("tanganke/stanford_cars", split=split)
    idx = random.Random(seed).sample(range(len(ds)), n)
    for i in idx:
        ex = ds[i]
        yield cv2.cvtColor(np.asarray(ex["image"].convert("RGB")), cv2.COLOR_RGB2BGR), ex["label"]


class Reader(CalibrationDataReader):
    def __init__(self, name, arrays):
        self.it = iter([{name: a} for a in arrays])

    def get_next(self):
        return next(self.it, None)


# ------------------------------------------------------------------ export + quantize
def export_and_quantize():
    from ultralytics import YOLO

    # Detector
    det_fp32 = Path(YOLO(str(MODELS / "yolov8n.pt")).export(format="onnx", imgsz=DET_IMGSZ, simplify=True,
                                                           dynamic=False, device="cpu"))
    det_pre = MODELS / "yolov8n_pre.onnx"
    quant_pre_process(str(det_fp32), str(det_pre))
    graph = onnx.load(str(det_pre)).graph
    # Keep the box-decoding tail of the Detect head (DFL softmax, anchors, concat) in FP32: it mixes
    # pixel coordinates with class scores in one tensor, which one INT8 scale can't represent well
    head = [n.name for n in graph.node if n.name.startswith("/model.22/") and n.op_type != "Conv"]
    det_in = graph.input[0].name
    calib = [letterbox(f) for f in video_frames(step=8, limit=60)]
    quantize_static(str(det_pre), str(MODELS / "yolov8n_int8.onnx"), Reader(det_in, calib),
                    quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8, calibrate_method=CalibrationMethod.Percentile,
                    nodes_to_exclude=head, extra_options={"CalibPercentile": 99.99})
    # Ultralytics reads class names, stride and image size from the ONNX metadata
    src, dst = onnx.load(str(det_fp32)), onnx.load(str(MODELS / "yolov8n_int8.onnx"))
    del dst.metadata_props[:]
    dst.metadata_props.extend(src.metadata_props)
    onnx.save(dst, str(MODELS / "yolov8n_int8.onnx"))
    det_pre.unlink()

    # Classifier
    names = json.loads((MODELS / "class_names.json").read_text(encoding="utf-8"))
    model = build_model(len(names))
    model.load_state_dict(torch.load(MODELS / "car_classifier.pt", map_location="cpu", weights_only=True))
    model.eval()
    cls_fp32 = MODELS / "car_classifier.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), str(cls_fp32), input_names=["images"],
                      output_names=["logits"], dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    cls_pre = MODELS / "car_classifier_pre.onnx"
    quant_pre_process(str(cls_fp32), str(cls_pre))
    calib = [CarClassifier._preprocess(img)[None] for img, _ in stanford("train", 300, seed=1)]
    quantize_static(str(cls_pre), str(MODELS / "car_classifier_int8.onnx"), Reader("images", calib),
                    quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8, calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"CalibPercentile": 99.99})
    cls_pre.unlink()


# ------------------------------------------------------------------ benchmark helpers
def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def latency_ms(fn, x, warmup=5, runs=40):
    for _ in range(warmup):
        fn(x)
    t = time.perf_counter()
    for _ in range(runs):
        fn(x)
    return (time.perf_counter() - t) / runs * 1000


def mb(path):
    return round(Path(path).stat().st_size / 1e6, 1)


def boxes_from(results):
    b = results.boxes
    keep = np.isin(b.cls.cpu().numpy().astype(int), list(VEHICLE_CLASSES))
    return b.xyxy.cpu().numpy()[keep]


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def agreement(ref, test):
    """Recall / precision of `test` boxes against `ref` boxes, IoU >= 0.5, greedy matching."""
    tp = n_ref = n_test = 0
    for r, t in zip(ref, test):
        n_ref, n_test = n_ref + len(r), n_test + len(t)
        if len(r) and len(t):
            m = iou(r, t)
            while m.size and m.max() >= 0.5:
                i, j = np.unravel_index(m.argmax(), m.shape)
                tp += 1
                m[i, :], m[:, j] = 0, 0
    return tp / max(n_ref, 1), tp / max(n_test, 1)


# ------------------------------------------------------------------ benchmark
def bench():
    from ultralytics import YOLO

    res = {"threads": THREADS, "cpu": {}, "gpu": {}}
    frames = video_frames(step=4, limit=120)

    # --- detector, CPU
    x = letterbox(frames[0])
    det = {}
    pt = YOLO(str(MODELS / "yolov8n.pt"))
    pt_preds = lambda imgs, **kw: [boxes_from(pt.predict(f, imgsz=DET_IMGSZ, conf=0.35, device="cpu", verbose=False, **kw)[0]) for f in imgs]  # noqa: E731
    xt = torch.from_numpy(x)
    with torch.inference_mode():
        det["PyTorch FP32"] = {"ms": latency_ms(lambda a: pt.model(a), xt), "mb": mb(MODELS / "yolov8n.pt")}
    ref = pt_preds(frames)
    for label, path in [("ONNX FP32", MODELS / "yolov8n.onnx"), ("ONNX INT8", MODELS / "yolov8n_int8.onnx")]:
        s = session(path)
        name = s.get_inputs()[0].name
        y = YOLO(str(path), task="detect")
        preds = [boxes_from(y.predict(f, imgsz=DET_IMGSZ, conf=0.35, device="cpu", verbose=False)[0]) for f in frames]
        rec, prec = agreement(ref, preds)
        det[label] = {"ms": latency_ms(lambda a: s.run(None, {name: a}), x), "mb": mb(path),
                      "recall_vs_fp32": rec, "precision_vs_fp32": prec}
    det["PyTorch FP32"].update(recall_vs_fp32=1.0, precision_vs_fp32=1.0)
    res["cpu"]["detector"] = det

    # --- classifier, CPU
    test = list(stanford("test", N_TEST, seed=0))
    xb = np.stack([CarClassifier._preprocess(img) for img, _ in test]).astype(np.float32)
    labels = np.array([l for _, l in test])
    names = json.loads((MODELS / "class_names.json").read_text(encoding="utf-8"))
    m = build_model(len(names))
    m.load_state_dict(torch.load(MODELS / "car_classifier.pt", map_location="cpu", weights_only=True))
    m.eval()

    def topk(logits):
        top5 = np.argsort(-logits, axis=1)[:, :5]
        return float((top5[:, 0] == labels).mean()), float((top5 == labels[:, None]).any(1).mean())

    cls = {}
    with torch.inference_mode():
        logits = np.concatenate([m(torch.from_numpy(xb[i:i + 64])).numpy() for i in range(0, len(xb), 64)])
        t1, t5 = topk(logits)
        cls["PyTorch FP32"] = {"ms": latency_ms(lambda a: m(a), torch.from_numpy(xb[:1])),
                               "mb": mb(MODELS / "car_classifier.pt"), "top1": t1, "top5": t5}
        try:  # FP16 on a CPU: most CPUs have no fast FP16 maths, so expect this to be slow
            mh = build_model(len(names)).eval()
            mh.load_state_dict(m.state_dict())
            mh = mh.half()
            cls["PyTorch FP16 (CPU)"] = {"ms": latency_ms(lambda a: mh(a), torch.from_numpy(xb[:1]).half(), runs=10)}
        except Exception as e:  # noqa: BLE001
            cls["PyTorch FP16 (CPU)"] = {"error": str(e)[:120]}
    for label, path in [("ONNX FP32", MODELS / "car_classifier.onnx"), ("ONNX INT8", MODELS / "car_classifier_int8.onnx")]:
        s = session(path)
        logits = np.concatenate([s.run(None, {"images": xb[i:i + 64]})[0] for i in range(0, len(xb), 64)])
        t1, t5 = topk(logits)
        cls[label] = {"ms": latency_ms(lambda a: s.run(None, {"images": a}), xb[:1]), "mb": mb(path),
                      "top1": t1, "top5": t5}
    res["cpu"]["classifier"] = cls

    # --- GPU: FP32 vs FP16 (what the app uses on an NVIDIA GPU)
    if torch.cuda.is_available() and torch.cuda.device_count():
        torch.backends.cudnn.benchmark = False
        g = {}
        xg = torch.from_numpy(xb[:1]).cuda()
        mg = m.cuda()
        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False):
            sync = lambda f: (lambda a: (f(a), torch.cuda.synchronize()))  # noqa: E731
            g["classifier FP32"] = {"ms": latency_ms(sync(mg), xg, runs=100)}
            logits = np.concatenate([mg(torch.from_numpy(xb[i:i + 64]).cuda()).float().cpu().numpy()
                                     for i in range(0, len(xb), 64)])
            g["classifier FP32"]["top1"], g["classifier FP32"]["top5"] = topk(logits)
            mg = mg.half()
            g["classifier FP16"] = {"ms": latency_ms(sync(mg), xg.half(), runs=100)}
            logits = np.concatenate([mg(torch.from_numpy(xb[i:i + 64]).cuda().half()).float().cpu().numpy()
                                     for i in range(0, len(xb), 64)])
            g["classifier FP16"]["top1"], g["classifier FP16"]["top5"] = topk(logits)
        for half in (False, True):
            y = YOLO(str(MODELS / "yolov8n.pt"))
            kw = dict(imgsz=640, conf=0.35, device="cuda", verbose=False, half=half)
            for f in frames[:10]:
                y.predict(f, **kw)
            t = time.perf_counter()
            preds = [boxes_from(y.predict(f, **kw)[0]) for f in frames]
            ms = (time.perf_counter() - t) / len(frames) * 1000
            g[f"detector {'FP16' if half else 'FP32'} (640, incl. pre/post)"] = {"ms": ms}
        res["gpu"] = g

    out = MODELS / "quantization_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="benchmark only")
    if not ap.parse_args().bench:
        export_and_quantize()
    bench()
