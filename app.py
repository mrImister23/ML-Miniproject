"""AutoVision AI - Streamlit front-end.

Run with:  streamlit run app.py
"""

import tempfile
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from autovision import AutoVisionPipeline, PipelineConfig

st.set_page_config(page_title="AutoVision AI", page_icon="🚗", layout="wide")


@st.cache_resource(show_spinner="Loading make & model classifier...")
def get_classifier(weights_mtime: float):
    """Cached per weights-file timestamp, so a retrained model is picked up automatically."""
    return AutoVisionPipeline.load_classifier()


def vehicles_table(vehicles) -> pd.DataFrame:
    rows = [
        {
            "ID": v.track_id,
            "Vehicle": v.display_name,
            "Body": v.spec.body if v.spec else "-",
            "Origin": v.spec.country if v.spec else "-",
            "Confidence": f"{v.confidence * 100:.0f}%" + (" (guess)" if v.uncertain else "") if v.spec else "-",
            "Distance (m)": round(v.distance_m, 1),
            "Proximity": v.proximity,
        }
        for v in sorted(vehicles, key=lambda r: r.distance_m)
    ]
    return pd.DataFrame(rows)


def render_side_panel(placeholder, vehicles, fps):
    with placeholder.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("FPS", f"{fps:.1f}")
        c2.metric("Vehicles", len(vehicles))
        c3.metric("Confident", sum(1 for v in vehicles if v.spec and not v.uncertain))
        if vehicles:
            st.dataframe(vehicles_table(vehicles), hide_index=True, use_container_width=True)
            recognised = [v for v in vehicles if v.spec]
            if recognised:
                v = min(recognised, key=lambda r: r.distance_m)
                s = v.spec
                guess = " · best guess" if v.uncertain else ""
                st.markdown(f"#### Closest: {s.make} {s.model} ({s.year}){guess}")
                st.markdown(
                    f"- **Body:** {s.body} · {s.category}\n"
                    f"- **Doors / seats:** {s.doors} / {s.seats}\n"
                    f"- **Origin:** {s.country}\n"
                    f"- **Distance:** {v.distance_m:.1f} m ({v.proximity})\n"
                    f"- **Confidence:** {v.confidence * 100:.0f}%"
                )
        else:
            st.caption("No vehicles in view.")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🚗 AutoVision AI")
st.sidebar.caption("Real-time vehicle make, model & spec recognition")

source = st.sidebar.radio("Input source", ["Webcam", "Video file", "Image"])

st.sidebar.subheader("Settings")
config = PipelineConfig(
    det_conf=st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05),
    cls_conf=st.sidebar.slider("Confident above", 0.05, 0.9, 0.25, 0.05,
                               help="Below this the most likely car is still shown, marked with '?' as a best guess."),
    classify_every=st.sidebar.slider("Classify each car every N frames", 1, 15, 5,
                                     help="Higher = faster FPS, slower label updates."),
    hfov_deg=st.sidebar.slider("Camera field of view (°)", 40, 120, 70,
                               help="Used for distance estimation. Laptop webcams ≈ 60-70°, dashcams ≈ 100-120°."),
    show_distance=st.sidebar.toggle("Proximity overlay", True),
    show_specs=st.sidebar.toggle("Spec HUD panel", True),
)

weights = Path(__file__).parent / "models" / "car_classifier.pt"
classifier = get_classifier(weights.stat().st_mtime if weights.exists() else 0.0)

st.title("AutoVision AI: Real-Time Vehicle Make, Model and Spec Recognition")
st.caption("YOLOv8 detection + ByteTrack tracking + MobileNetV3 fine-grained classification (Stanford Cars, 196 classes)")

if classifier is None:
    get_classifier.clear()  # retry on the next rerun, e.g. once training has finished
    st.warning("Make/model classifier not found in `models/`. Run `train_classifier.ipynb` first. "
               "Running in detection-only mode for now.")

left, right = st.columns([3, 2])
side = right.empty()

# ---------------------------------------------------------------- webcam
if source == "Webcam":
    if "webcam_pipe" not in st.session_state:
        st.session_state.webcam_pipe = AutoVisionPipeline(classifier=classifier)
        st.session_state.webcam_shared = {"vehicles": [], "lock": threading.Lock()}
    pipe = st.session_state.webcam_pipe
    pipe.config = config
    shared = st.session_state.webcam_shared

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        out, vehicles = pipe.process_frame(img)
        with shared["lock"]:
            shared["vehicles"] = vehicles
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    with left:
        ctx = webrtc_streamer(
            key="autovision-webcam",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        st.caption("Tip: point the webcam at a street, or at car photos on another screen.")

    while ctx.state.playing:
        with shared["lock"]:
            vehicles = list(shared["vehicles"])
        render_side_panel(side, vehicles, pipe.fps)
        time.sleep(0.5)
    render_side_panel(side, [], 0.0)

# ---------------------------------------------------------------- video file
elif source == "Video file":
    upload = left.file_uploader("Upload a street-view / dashcam video", type=["mp4", "avi", "mov", "mkv"])
    samples = sorted(p for p in (Path(__file__).parent / "samples").glob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
    sample = left.selectbox("...or pick a sample from `samples/`", ["-"] + [p.name for p in samples]) if samples else "-"
    save_output = left.checkbox("Save annotated video for download", value=False)

    video_path = None
    if upload is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix)
        tmp.write(upload.read())
        tmp.close()
        video_path = tmp.name
    elif sample != "-":
        video_path = str(Path(__file__).parent / "samples" / sample)

    if video_path and left.button("▶ Run AutoVision", type="primary"):
        pipe = AutoVisionPipeline(classifier=classifier, config=config)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_slot = left.empty()
        progress = left.progress(0.0, text="Processing...")
        writer, out_path = None, None
        if save_output:
            out_path = str(Path(tempfile.gettempdir()) / f"autovision_{int(time.time())}.mp4")

        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Keep large videos fast: process at most 1280 px wide
            if frame.shape[1] > 1280:
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
            out, vehicles = pipe.process_frame(frame)
            if save_output:
                if writer is None:
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (out.shape[1], out.shape[0]))
                writer.write(out)
            frame_slot.image(out, channels="BGR", use_container_width=True)
            if i % 10 == 0:
                render_side_panel(side, vehicles, pipe.fps)
                progress.progress(min(i / total, 1.0), text=f"Frame {i}/{total}")
            i += 1

        cap.release()
        progress.progress(1.0, text=f"Done: {i} frames, avg {pipe.fps:.1f} FPS")
        if writer is not None:
            writer.release()
            with open(out_path, "rb") as f:
                left.download_button("⬇ Download annotated video", f, file_name="autovision_output.mp4", mime="video/mp4")
    elif not video_path:
        left.info("Upload a video to start.")

# ---------------------------------------------------------------- image
else:
    upload = left.file_uploader("Upload a photo with cars", type=["jpg", "jpeg", "png", "webp"])
    if upload is not None:
        img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
        pipe = AutoVisionPipeline(classifier=classifier, config=config)
        out, vehicles = pipe.process_frame(img)
        left.image(out, channels="BGR", use_container_width=True)
        render_side_panel(side, vehicles, pipe.fps)
    else:
        left.info("Upload an image to start.")
