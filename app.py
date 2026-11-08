"""AutoVision AI - Streamlit front-end.

Run with:  streamlit run app.py
"""

import queue
import shutil
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from autovision import AutoVisionPipeline, PipelineConfig
from autovision.cpu import limit_cpu_threads

CPU_INFO = limit_cpu_threads()  # before any model runs

st.set_page_config(page_title="AutoVision AI", page_icon="🚗", layout="wide")

# Clean, minimal look: quiet header, generous spacing, thin dividers, lighter headings
st.markdown(
    """
    <style>
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { border-right: 1px solid #E6D9BC; }
    .block-container { padding-top: 2.5rem; max-width: 1400px; }
    h1 { font-weight: 600; letter-spacing: -0.02em; }
    h4 { font-weight: 600; }
    [data-testid="stMetric"] { background: #FFFDF7; border: 1px solid #E6D9BC; border-radius: 8px; padding: 0.6rem 0.9rem; }
    [data-testid="stMetricValue"] { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading make & model classifier...")
def get_classifier(weights_mtime: float):
    """Cached per weights-file timestamp, so a retrained model is picked up automatically."""
    return AutoVisionPipeline.load_classifier()


def to_jpeg(frame_bgr: np.ndarray, max_width: int = 960, quality: int = 80) -> bytes:
    """Small, fast JPEG for the browser. Streamlit's own conversion is a slow quality-100 encode
    of the full frame (~3x bigger), which is what made playback stutter."""
    if frame_bgr.shape[1] > max_width:
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(frame_bgr.shape[0] * max_width / frame_bgr.shape[1])),
                               interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def start_worker(cap: cv2.VideoCapture, pipe: AutoVisionPipeline, out_path: str | None, fps: float,
                 max_width: int = 1280, jpeg_width: int = 960, jpeg_quality: int = 80) -> queue.Queue:
    """Decode, run the AI and encode frames in a background thread, so that work overlaps with
    Streamlit sending the previous frame to the browser. Yields (jpeg, vehicles, fps) items, then
    None at the end of the video (or the exception, if one happened)."""
    results: queue.Queue = queue.Queue(maxsize=8)
    results.stats = {"decode": 0.0, "ai": 0.0, "jpeg": 0.0}  # total seconds per stage, shown at the end

    to_save: queue.Queue = queue.Queue(maxsize=64)

    def save():  # MP4 encoding is slow, so it gets its own thread too
        container = stream = None
        while (out := to_save.get()) is not None:
            if container is None:
                # H.264 (not OpenCV's mp4v) so browsers can play the result; faststart lets playback
                # begin before the whole file has downloaded
                container = av.open(out_path, "w", options={"movflags": "faststart"})
                stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
                stream.width, stream.height = out.shape[1] // 2 * 2, out.shape[0] // 2 * 2  # yuv420p needs even
                stream.pix_fmt = "yuv420p"
                stream.options = {"preset": "veryfast", "crf": "23"}
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(out[:stream.height, :stream.width]), format="bgr24")
            container.mux(stream.encode(frame))
        if container is not None:
            container.mux(stream.encode(None))
            container.close()

    def run():
        saver = threading.Thread(target=save, daemon=True) if out_path else None
        if saver:
            saver.start()
        try:
            while True:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] > max_width:  # keep large videos fast
                    frame = cv2.resize(frame, (max_width, int(frame.shape[0] * max_width / frame.shape[1])))
                t1 = time.perf_counter()
                out, vehicles = pipe.process_frame(frame)
                t2 = time.perf_counter()
                if saver:
                    to_save.put(out)
                jpeg = to_jpeg(out, jpeg_width, jpeg_quality)
                stats = results.stats
                stats["decode"] += t1 - t0
                stats["ai"] += t2 - t1
                stats["jpeg"] += time.perf_counter() - t2
                # Times out only if the run was stopped or the page rerun, i.e. nobody is reading
                results.put((jpeg, vehicles, pipe.fps), timeout=10)
            if saver:
                to_save.put(None)
                saver.join()  # finish the file before telling the app it can offer the download
            results.put(None)
        except queue.Full:
            pass
        except Exception as e:  # show it in the app instead of silently hanging
            results.put(e)
        finally:
            cap.release()
            if saver and saver.is_alive():
                to_save.put(None)

    threading.Thread(target=run, daemon=True).start()
    return results


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
        c1.metric("FPS", f"{fps:.0f}")
        c2.metric("Vehicles", len(vehicles))
        c3.metric("Confident", sum(1 for v in vehicles if v.spec and not v.uncertain))
        if vehicles:
            st.dataframe(vehicles_table(vehicles), hide_index=True, width="stretch")
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
st.sidebar.title("AutoVision AI")
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

st.title("Vehicle Make, Model & Spec Recognition")
st.caption("YOLOv8 detection · ByteTrack tracking · MobileNetV3 classification · Stanford Cars, 196 classes")
st.divider()

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
    has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    live = left.toggle("Live preview", value=has_gpu,
                       help="Show frames while they're processed. Without a GPU (e.g. on Streamlit Cloud) this "
                            "lags, so by default the video is processed first and then played smoothly.")
    save_output = live and left.checkbox("Save annotated video for download", value=False)

    video_path = None
    if upload is not None:
        # Save each upload to disk once, in chunks. Re-reading the whole file into memory on every
        # rerun made big videos hang the app on Streamlit Cloud (about 1 GB of RAM there).
        saved = st.session_state.get("uploaded_video")
        if saved and saved[0] == upload.file_id and Path(saved[1]).exists():
            video_path = saved[1]
        else:
            if saved:
                Path(saved[1]).unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
                upload.seek(0)
                shutil.copyfileobj(upload, tmp, 1024 * 1024)
            video_path = tmp.name
            st.session_state.uploaded_video = (upload.file_id, video_path)
    elif sample != "-":
        video_path = str(Path(__file__).parent / "samples" / sample)

    if video_path and left.button("▶ Run AutoVision", type="primary"):
        pipe = AutoVisionPipeline(classifier=classifier, config=config)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened() or not cap.read()[0]:
            cap.release()
            left.error("Can't read this video. Phone videos are often HEVC/H.265, which OpenCV can't decode "
                       "here. Convert it to H.264 MP4 (e.g. with HandBrake or an online converter) and try again.")
            st.stop()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        note = ""
        if pipe.detector.device == "cpu":
            reported, usable, quota = CPU_INFO
            note = f" No GPU here: running on {usable} CPU core(s) (server reports {reported}" + (
                f", limit {quota:g})." if quota else ").")
        left.caption(f"{w}×{h}, {src_fps:.0f} FPS, {total} frames. The first frame takes a few seconds while the models warm up.{note}")
        frame_slot = left.empty()
        progress = left.progress(0.0, text="Processing...")
        out_path = str(Path(tempfile.gettempdir()) / f"autovision_{int(time.time())}.mp4") if save_output or not live else None
        if not live:
            left.info("Processing the whole video first, then it plays smoothly below. The preview updates about once a second.")

        if pipe.detector.device == "cpu":  # CPU-only server, e.g. Streamlit Community Cloud
            # Smaller frames to process, and smaller JPEGs (every frame goes over the internet;
            # a busy 4K street scene is ~65 KB a frame at 960 px, ~29 KB at 720 px)
            results = start_worker(cap, pipe, out_path, src_fps, max_width=960, jpeg_width=720, jpeg_quality=70)
        else:
            results = start_worker(cap, pipe, out_path, src_fps)
        i = 0
        t_start = t_first = time.perf_counter()
        next_panel = 0.0
        waited = sent = 0.0
        while True:
            t0 = time.perf_counter()
            item = results.get()
            waited += time.perf_counter() - t0
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            jpeg, vehicles, fps = item

            if live:
                # Play at the video's real speed (never faster), so the browser isn't flooded with frames
                # it can't draw in time, which is what built up a backlog and froze the page.
                ahead = t_start + i / src_fps - time.perf_counter()
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < -0.25:  # fell behind (e.g. GPU warm-up): re-sync instead of racing to catch up
                    t_start -= ahead
                t0 = time.perf_counter()
                frame_slot.image(jpeg, width="stretch")
                sent += time.perf_counter() - t0

            now = time.perf_counter()
            if now >= next_panel:  # the table is heavy for the browser, so refresh it ~once a second
                if not live:
                    frame_slot.image(jpeg, width="stretch")
                render_side_panel(side, vehicles, fps)
                progress.progress(min(i / total, 1.0), text=f"Frame {i}/{total}")
                next_panel = now + 1.0
            i += 1

        elapsed = time.perf_counter() - t_first
        progress.progress(1.0, text=f"Done: {i} frames in {elapsed:.0f} s ({i / max(elapsed, 1e-6):.1f} FPS)")
        ms = {k: 1000 * v / max(i, 1) for k, v in results.stats.items()}
        left.caption(f"Time per frame: decode {ms['decode']:.0f} ms · AI {ms['ai']:.0f} ms · JPEG {ms['jpeg']:.0f} ms "
                     f"· waiting for the AI {1000 * waited / max(i, 1):.0f} ms · sending {1000 * sent / max(i, 1):.0f} ms")
        if not live and out_path and Path(out_path).exists():
            frame_slot.video(out_path, autoplay=True, muted=True)
        if out_path and Path(out_path).exists():
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
        left.image(out, channels="BGR", width="stretch")
        render_side_panel(side, vehicles, pipe.fps)
    else:
        left.info("Upload an image to start.")
