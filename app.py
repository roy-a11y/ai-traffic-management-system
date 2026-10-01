from __future__ import annotations

import math
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# COCO class ids -> names
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

MIN_HITS = 3            # a track must be seen this many times to count (kills ID flicker)
STOP_THRESHOLD = 0.25   # moved < 25% of its own box size over ~1 s => "stopped"
CYCLE_SECONDS = 100     # fixed signal cycle so timings are comparable
YELLOW_SECONDS = 5


@dataclass
class TrafficResult:
    unique_vehicles: int
    vehicle_counts: dict[str, int]
    in_view: list[int]
    stopped: list[int]
    avg_in_view: float
    peak_in_view: int
    stopped_ratio: float
    congestion: str
    timing: dict[str, int]
    preview_frames: list[np.ndarray]
    analyzed_seconds: float
    video_seconds: float
    truncated: bool
    per_minute: float


st.set_page_config(page_title="AI Traffic Management System", layout="wide")


def congestion_level(avg_in_view: float, stopped_ratio: float) -> str:
    # Heuristic thresholds. They depend on camera angle/zoom; tune per camera.
    if avg_in_view >= 18 or (avg_in_view >= 5 and stopped_ratio >= 0.6):
        return "High"
    if avg_in_view >= 8 or (avg_in_view >= 3 and stopped_ratio >= 0.4):
        return "Medium"
    return "Low"


def suggest_signal_timing(level: str) -> dict[str, int]:
    green = {"Low": 25, "Medium": 40, "High": 60}[level]
    return {
        "Green": green,
        "Yellow": YELLOW_SECONDS,
        "Red": CYCLE_SECONDS - green - YELLOW_SECONDS,
    }


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def analyze_video(
    video_path: Path,
    model_name: str,
    confidence: float,
    frame_skip: int,
    max_frames: int,
    imgsz: int,
) -> TrafficResult:
    # Fresh model per run: ByteTrack state must not leak between videos.
    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not read the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    planned = min(max_frames, math.ceil(total_frames / frame_skip)) if total_frames > 0 else max_frames
    planned = max(planned, 1)
    preview_at = set(np.linspace(0, planned - 1, min(6, planned)).astype(int).tolist())

    window = max(3, round(fps / frame_skip))  # ~1 second of observations
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    hits: Counter = Counter()
    classes: dict[int, Counter] = defaultdict(Counter)

    in_view: list[int] = []
    stopped: list[int] = []
    judged_total = 0
    stopped_total = 0
    previews: list[np.ndarray] = []

    processed = 0
    frame_index = 0
    truncated = False

    while True:
        if processed >= max_frames:
            truncated = bool(cap.grab())  # any frames left?
            break
        if not cap.grab():  # grab() skips decoding; cheap for skipped frames
            break

        if frame_index % frame_skip == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break

            res = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                imgsz=imgsz,
                classes=list(COCO_VEHICLES),
                verbose=False,
            )[0]

            draw = frame.copy() if processed in preview_at else None
            n_view = 0
            n_stop = 0

            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.int().cpu().tolist()
                cls_ids = res.boxes.cls.int().cpu().tolist()
                boxes = res.boxes.xyxy.cpu().numpy()

                for tid, cid, (x1, y1, x2, y2) in zip(ids, cls_ids, boxes):
                    hits[tid] += 1
                    classes[tid][COCO_VEHICLES[cid]] += 1
                    if hits[tid] < MIN_HITS:
                        continue

                    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    diag = math.hypot(x2 - x1, y2 - y1)
                    track = history[tid]
                    track.append((center, diag))
                    n_view += 1

                    state = "new"
                    if len(track) == window:
                        mean_diag = float(np.mean([d for _, d in track]))
                        disp = float(np.linalg.norm(track[-1][0] - track[0][0]))
                        is_stopped = disp / max(mean_diag, 1e-6) < STOP_THRESHOLD
                        judged_total += 1
                        if is_stopped:
                            stopped_total += 1
                            n_stop += 1
                        state = "stopped" if is_stopped else "moving"

                    if draw is not None:
                        color = {"stopped": (0, 0, 255), "moving": (0, 200, 0), "new": (200, 200, 200)}[state]
                        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                        cv2.rectangle(draw, p1, p2, color, 2)
                        cv2.putText(
                            draw,
                            f"#{tid} {COCO_VEHICLES[cid]}",
                            (p1[0], max(20, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

            in_view.append(n_view)
            stopped.append(n_stop)
            if draw is not None:
                previews.append(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
            processed += 1

        frame_index += 1

    cap.release()

    # Final class per track = its most frequent label; only confirmed tracks count.
    vehicle_counts = {name: 0 for name in COCO_VEHICLES.values()}
    for tid, n in hits.items():
        if n >= MIN_HITS:
            vehicle_counts[classes[tid].most_common(1)[0][0]] += 1
    unique = sum(vehicle_counts.values())

    analyzed_seconds = frame_index / fps
    video_seconds = total_frames / fps if total_frames > 0 else analyzed_seconds
    avg_in_view = float(np.mean(in_view)) if in_view else 0.0
    stopped_ratio = stopped_total / judged_total if judged_total else 0.0
    level = congestion_level(avg_in_view, stopped_ratio)

    return TrafficResult(
        unique_vehicles=unique,
        vehicle_counts=vehicle_counts,
        in_view=in_view,
        stopped=stopped,
        avg_in_view=avg_in_view,
        peak_in_view=max(in_view) if in_view else 0,
        stopped_ratio=stopped_ratio,
        congestion=level,
        timing=suggest_signal_timing(level),
        preview_frames=previews,
        analyzed_seconds=analyzed_seconds,
        video_seconds=video_seconds,
        truncated=truncated,
        per_minute=unique / (analyzed_seconds / 60) if analyzed_seconds > 0 else 0.0,
    )


def plot_vehicle_counts(counts: dict[str, int]):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(counts), list(counts.values()), color=["#2e86de", "#10ac84", "#ff9f43", "#ee5253"])
    ax.set_title("Unique Vehicles by Type")
    ax.set_xlabel("Vehicle type")
    ax.set_ylabel("Unique vehicles")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_over_time(in_view: list[int], stopped: list[int]):
    x = range(1, len(in_view) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, in_view, color="#5f27cd", linewidth=2, label="In view")
    ax.plot(x, stopped, color="#ee5253", linewidth=2, label="Stopped")
    ax.fill_between(x, in_view, alpha=0.15, color="#5f27cd")
    ax.set_title("Vehicles In View vs Stopped")
    ax.set_xlabel("Processed frame")
    ax.set_ylabel("Vehicles")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def congestion_badge(level: str) -> str:
    colors = {"Low": "#10ac84", "Medium": "#ff9f43", "High": "#ee5253"}
    return f"<span style='color:{colors[level]}; font-weight:700'>{level}</span>"


# ---------------------------------------------------------------- UI

st.title("AI Traffic Management System")
st.caption("Upload a traffic video, track vehicles, estimate congestion, and get a simple signal timing suggestion.")

with st.sidebar:
    st.header("Analysis Settings")
    model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    imgsz = st.select_slider("Inference size", [640, 960, 1280], value=640)
    confidence = st.slider("YOLO confidence", 0.10, 0.90, 0.35, 0.05)
    frame_skip = st.slider("Analyze every Nth frame", 1, 10, 2)
    max_frames = st.slider("Maximum processed frames", 10, 1000, 200, 10)
    st.caption("Keep the frame skip low: tracking loses vehicles when frames are far apart.")

uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is None:
    st.info("Upload a traffic video to start the analysis.")
    st.stop()

# Save the upload once per file, not on every rerun; delete the previous one.
upload_key = (uploaded.name, uploaded.size)
if st.session_state.get("upload_key") != upload_key:
    old = st.session_state.get("video_path")
    if old:
        Path(old).unlink(missing_ok=True)
    st.session_state["upload_key"] = upload_key
    st.session_state["video_path"] = str(save_upload(uploaded))
    st.session_state["result"] = None

video_path = Path(st.session_state["video_path"])

if video_path.suffix.lower() in {".mp4", ".webm"}:
    st.video(str(video_path))
else:
    st.caption(f"Preview unavailable for {video_path.suffix} files in the browser; analysis still works.")

if st.button("Analyze Traffic", type="primary"):
    with st.spinner("Tracking vehicles and calculating congestion..."):
        try:
            st.session_state["result"] = analyze_video(
                video_path, model_name, confidence, frame_skip, max_frames, imgsz
            )
        except Exception as exc:
            st.session_state["result"] = None
            st.error(f"Analysis failed: {exc}")

result: TrafficResult | None = st.session_state.get("result")

if result is not None:
    if result.truncated:
        st.warning(
            f"Only the first {result.analyzed_seconds:.0f}s of a {result.video_seconds:.0f}s video were analyzed. "
            "Raise 'Maximum processed frames' or the frame skip to cover the rest."
        )

    st.subheader("Traffic Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Unique vehicles", result.unique_vehicles)
    c2.metric("Vehicles/minute", f"{result.per_minute:.1f}")
    c3.metric("Avg in view", f"{result.avg_in_view:.1f}")
    c4.metric("Stopped share", f"{result.stopped_ratio:.0%}")
    c5.markdown(f"**Congestion**<br>{congestion_badge(result.congestion)}", unsafe_allow_html=True)
    st.caption(
        "Unique vehicles = distinct tracked IDs seen in at least "
        f"{MIN_HITS} frames. Congestion uses fixed thresholds on density and stopped share; "
        "tune them per camera. A stopped queue is normal at a red light, so judge over a full signal cycle."
    )

    st.subheader("Suggested Signal Timing")
    for col, (name, secs) in zip(st.columns(3), result.timing.items()):
        col.metric(name, f"{secs} sec")
    st.caption(f"Single-approach heuristic on a fixed {CYCLE_SECONDS}s cycle. Real timing needs per-direction counts.")

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_vehicle_counts(result.vehicle_counts), clear_figure=True)
    with right:
        st.pyplot(plot_over_time(result.in_view, result.stopped), clear_figure=True)

    st.subheader("Vehicle Counts")
    st.dataframe(
        pd.DataFrame([{"Vehicle Type": k.title(), "Unique vehicles": v} for k, v in result.vehicle_counts.items()]),
        hide_index=True,
    )

    if result.preview_frames:
        st.subheader("Annotated Sample Frames (spread across the video)")
        st.caption("Red = stopped, green = moving, grey = not enough history yet.")
        cols = st.columns(3)
        for i, frame in enumerate(result.preview_frames):
            cols[i % 3].image(frame, caption=f"Sample {i + 1}")

