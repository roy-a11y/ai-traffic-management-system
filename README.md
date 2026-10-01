# AI Traffic Management System

> Real-time vehicle detection, tracking, and congestion analysis from traffic video — with adaptive signal timing suggestions.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00d4aa?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5c3ee8?style=flat-square&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Overview

This project is a prototype traffic analysis dashboard. Upload a traffic video and the system will:

1. **Detect & classify** vehicles (car, motorcycle, bus, truck) using YOLOv8.
2. **Track** each vehicle across frames with ByteTrack, assigning persistent IDs.
3. **Estimate congestion** based on vehicle density and the fraction of stopped vehicles.
4. **Suggest signal timings** (green / yellow / red) adapted to the congestion level.

Everything runs inside an interactive **Streamlit** web app — no external services required.

---

## Demo

| Metric Cards | Annotated Frames |
|:---:|:---:|
| Live counts, vehicles/min, stopped share, congestion badge | Color-coded bounding boxes (red = stopped, green = moving) |

> Upload any `.mp4`, `.avi`, `.mov`, or `.mkv` traffic clip to try it yourself.

---

## How It Works

```
Video Upload
    │
    ▼
┌──────────────────────────┐
│  YOLOv8 Object Detection │  ← detects cars, motorcycles, buses, trucks
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  ByteTrack Multi-Object  │  ← assigns persistent IDs across frames
│       Tracking           │  ← minimum-hit filter suppresses flicker
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Congestion Estimation   │  ← density + stopped-vehicle ratio
│  + Signal Timing         │  ← heuristic on a 100s cycle
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Streamlit Dashboard     │  ← metrics, charts, annotated frames
└──────────────────────────┘
```

### Key Design Decisions

- **Unique vehicles** = distinct tracker IDs seen in ≥ 3 frames (not raw per-frame detections). This eliminates false positives from ID flicker.
- **Stopped detection** = a vehicle that moved less than 25% of its bounding-box diagonal over a ~1 second sliding window.
- **Congestion levels** (Low / Medium / High) are set by fixed thresholds on average vehicles in view and stopped share. These are meant to be tuned per camera angle.
- **Signal timing** is a single-approach heuristic splitting a fixed 100-second cycle into green, 5s yellow, and red. Real traffic control would need per-direction counts.

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | [YOLOv8](https://docs.ultralytics.com/) (Ultralytics) |
| Multi-Object Tracking | ByteTrack via Ultralytics |
| Video Processing | [OpenCV](https://opencv.org/) |
| Web Dashboard | [Streamlit](https://streamlit.io/) |
| Data Handling | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Visualization | [Matplotlib](https://matplotlib.org/) |

---

## Getting Started

### Prerequisites

- **Python 3.10+** (tested on 3.11 and 3.12)
- A traffic video file (`.mp4`, `.avi`, `.mov`, or `.mkv`)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/roy-a11y/ai-traffic-management-system.git
cd ai-traffic-management-system

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**.

---

## Usage

1. **Upload** a traffic video using the file uploader.
2. **Adjust settings** in the sidebar:
   - YOLO model (`yolov8n` for speed, `yolov8s/m` for accuracy)
   - Inference size (640 / 960 / 1280)
   - Confidence threshold
   - Frame skip interval
   - Maximum frames to process
3. Click **"Analyze Traffic"** and wait for the results.
4. Review the **Traffic Summary** (unique vehicles, vehicles/min, congestion level), **Signal Timing**, **charts**, and **annotated sample frames**.

---

## Project Structure

```
ai-traffic-management-system/
├── app.py                # Main application — detection, tracking, UI
├── requirements.txt      # Python dependencies
├── run_app.bat           # Windows quick-launch script
├── yolov8n.pt            # YOLOv8 nano weights (auto-downloaded if missing)
├── .gitignore
└── README.md
```

---

## Configuration & Tuning

| Parameter | Default | Notes |
|---|---|---|
| `MIN_HITS` | 3 | Minimum detections before a track is confirmed |
| `STOP_THRESHOLD` | 0.25 | Movement / box-diagonal ratio below which a vehicle is "stopped" |
| `CYCLE_SECONDS` | 100 | Total signal cycle length (seconds) |
| `YELLOW_SECONDS` | 5 | Fixed yellow phase duration |

Congestion thresholds are in `congestion_level()` — tune them for your camera's field of view.

---

## Limitations

- **Single camera, static video only** — not a real-time multi-camera system.
- **Frame skip degrades tracking** — keep "Analyze every Nth frame" at 1–3 for best results.
- Larger YOLO models (`yolov8s`, `yolov8m`) improve detection of small/distant vehicles but are slower. Non-bundled models download automatically on first use.
- Signal timing is a **single-approach heuristic** — real intersections need per-direction analysis.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Shamba Roy**
- GitHub: [@roy-a11y](https://github.com/roy-a11y)