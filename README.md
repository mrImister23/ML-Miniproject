# AutoVision AI: Real-Time Vehicle Make, Model and Spec Recognition

**Group AutoVision** | Smart Automotive & Connected ADAS

| Member | Student ID |
|---|---|
| Pasin Pengranai | 116710905061-3 |
| Siraphat Satthatham | 116710905062-1 |
| Chanathip Saexung | 116710905081-1 |

Presentation slides (Thai): [docs/AutoVision_AI_slides.pdf](docs/AutoVision_AI_slides.pdf)

A Streamlit app that detects vehicles in a **webcam stream, a street-view/dashcam video or a photo**, recognises each car's **make, model and year**, estimates its **distance**, and shows a **spec HUD overlay**.

## Pipeline

```
frame ─► YOLOv8n (car/bus/truck) ─► ByteTrack IDs ─► crop ─► MobileNetV3-Large (196 classes)
      ─► per-track smoothing (avg of last 10 predictions) ─► pinhole distance ─► HUD overlay
```

| Stage | Task | Model / method |
|---|---|---|
| Vehicle detection & tracking | Object detection | YOLOv8n (COCO pretrained) + ByteTrack |
| Make & model recognition | Fine-grained classification | MobileNetV3-Large fine-tuned on Stanford Cars |
| Proximity & distance | Monocular estimation | Pinhole model on box height |
| Specification HUD | Lookup | Class name parser + `data/specs.json` |

Speed tricks: each tracked car is only re-classified every *N* frames (slider, default 5), crops are batch-classified in fp16 on the GPU, and the label is a rolling average so it doesn't flicker.

## Project structure

```
app.py                  Streamlit UI (webcam / video / image)
train_classifier.ipynb  Fine-tunes MobileNetV3 on Stanford Cars
autovision/
  detector.py           YOLOv8 + ByteTrack
  classifier.py         MobileNetV3 make/model classifier
  tracker_state.py      per-track prediction smoothing
  distance.py           distance estimate + near/medium/far
  specs.py              "BMW M3 Coupe 2012" -> make/model/body/year/origin
  hud.py                OpenCV HUD drawing
  pipeline.py           ties everything together
data/specs.json         make origin + body-type info
models/                 trained weights + class names
```

## Setup (Windows, NVIDIA GPU)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt   # app + training notebook
```

To run only the app, `requirements.txt` is enough.

## Deploy on Streamlit Community Cloud

Point the app at `app.py`. `requirements.txt` installs CPU-only PyTorch and `packages.txt` adds the Linux system libraries OpenCV needs (`libgl1`, `libglib2.0-0`). The cloud has no GPU, so expect a few FPS on video; webcam mode may also need a TURN server to connect from some networks.

## 1. Train the classifier

Open `train_classifier.ipynb` with the `.venv` kernel and run all cells. It downloads Stanford Cars from Hugging Face (~1.9 GB), trains for 30 epochs (about 40-60 min on an RTX 3050) and writes `models/car_classifier.pt` and `models/class_names.json`.

## 2. Run the app

```bash
streamlit run app.py
```

- **Webcam**: click *START* and allow camera access.
- **Video file**: upload a dashcam / street video (or drop files in `samples/`), press *Run AutoVision*, optionally save the annotated video.
- **Image**: upload a photo.

Sidebar sliders control detection confidence, the "Confident above" threshold (below it the most likely car is still shown, marked `?` as a best guess), classify-every-N-frames, and the camera field of view used for distance.

## Results (RTX 3050 6GB Laptop, 30 epochs, ~70 min)

| Metric | Value |
|---|---|
| Top-1 accuracy (make + model + year, 196 classes) | **88.8%** |
| Top-5 accuracy | **97.6%** |
| Make-level accuracy (brand correct) | **94.5%** |
| End-to-end FPS on a 1280×720 video (detect + track + classify + HUD) | **31.4 FPS** |

Most errors are between near-identical vehicles, e.g. Dodge Sprinter vs Mercedes-Benz Sprinter, or the same model from a different year (Bentley Continental GT 2007 vs 2012).

## Quantization: FP32 vs FP16 vs INT8

`python tools/quantize_models.py` exports both models to ONNX, quantizes them to INT8 (ONNX Runtime
static post-training quantization, per-channel weights, calibrated on street frames and Stanford Cars
training images) and benchmarks them. Results in `models/quantization_results.json`. CPU numbers use
2 threads, like a Streamlit Cloud server; accuracy is on 2,000 Stanford Cars test images.

| Model | Format | Time / image | Size | Accuracy |
|---|---|---|---|---|
| Detector (YOLOv8n, 416 px, CPU) | PyTorch FP32 | 57.4 ms | 6.5 MB* | reference |
| | ONNX FP32 | 23.0 ms | 12.8 MB | 96% of FP32's boxes found |
| | **ONNX INT8** | **12.9 ms** | **3.6 MB** | 90% of FP32's boxes found, 95% precision |
| Classifier (MobileNetV3, CPU) | PyTorch FP32 | 13.7 ms | 18.0 MB | 87.4% Top-1 |
| | PyTorch FP16 | 469.8 ms | | (CPUs have no fast FP16) |
| | **ONNX FP32** | **3.7 ms** | 17.8 MB | 87.4% Top-1 |
| | ONNX INT8 | 3.9 ms | 5.0 MB | 86.4% Top-1 |
| Classifier (GPU, RTX 3050) | FP32 / FP16 | 8.3 / 8.0 ms | | 87.4% / 87.5% Top-1 |
| Detector (GPU, 640 px) | FP32 / FP16 | 9.1 / 9.6 ms | | |

\* Ultralytics stores `yolov8n.pt` in FP16 on disk; it runs as FP32 on a CPU.

What the app uses: on a GPU, PyTorch FP16 (FP16 and FP32 run at the same speed for models this
small). Without a GPU, the INT8 detector and the ONNX FP32 classifier, since INT8 made the
classifier no faster but lost 1% accuracy. End to end on 2 CPU cores that halves the time per frame
(18.7 → 9.3 ms).

## Evaluation metrics

| Metric | Where |
|---|---|
| Top-1 / Top-5 accuracy | Stanford Cars test split, printed at the end of the notebook (`models/metrics.json`) |
| mAP@0.5 | YOLOv8n COCO pretrained (37.3 mAP@0.5:0.95 reported by Ultralytics) |
| FPS | Shown live on the HUD and in the app |

## Limitations

- Stanford Cars covers 196 mostly US-market models from 1991-2012. Newer or Thai-market cars (e.g. Toyota Yaris Ativ, Isuzu D-Max) are not in the dataset, so they are shown as the nearest look-alike, usually marked `?` as a low-confidence best guess.
- Distance assumes typical vehicle heights and a known camera field of view, so it is an approximation.
- Specs are limited to what can be derived reliably (body type, doors/seats range, country of origin); engine and power figures are not included.
