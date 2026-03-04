# AURA – AI Driven Unified Real-Time Crowd & Resource Analytics System

AURA is a production-style AI safety analytics platform for crowded public spaces such as stadiums, concerts, railway stations, and festivals. It combines crowd detection, density risk scoring, stampede motion analysis, rolling video buffer intelligence, and lost-item retrieval using CLIP-based visual similarity.

## Features

- **Real-time video monitoring** (webcam/CCTV) at fixed **640x480** and configurable FPS (default ~6 FPS).
- **Crowd detection & counting** using **YOLOv8** (person class).
- **Crowd density estimation** with threshold-based high-density alerting.
- **Stampede risk detection** using frame differencing + mean movement score.
- **Rolling 5-minute frame buffer** using `deque` to support forensic search.
- **Lost & Found AI Search** using YOLO object crops + CLIP embeddings + cosine similarity.
- **Multi-channel alerting**:
  - Console alerts
  - Audio alarm
  - Twilio WhatsApp notification
- **Streamlit dashboard** modules:
  - LIVE MONITORING
  - LOG VIEWER
  - ANALYTICS
  - LOST & FOUND

---

## Project Structure

```text
aura_project/
│
├── main.py
├── detection.py
├── motion_analysis.py
├── buffer.py
├── lost_item_search.py
├── alerts.py
├── dashboard.py
├── utils.py
├── __init__.py
│
├── models/
│   └── README.md            # Place custom yolov8.pt here
│
└── logs/
    └── events.csv

app.py
requirements.txt
README.md
alert.wav
```

---

## Module Explanation

### `detection.py`
- Loads YOLOv8 model.
- Detects people (`class_id == 0`).
- Draws bounding boxes and confidences.
- Calculates:
  - `people_count`
  - `density = people_count / frame_area`
- Emits high-density flag if threshold exceeded.

### `motion_analysis.py`
- Converts current/previous frame to grayscale.
- Computes absolute difference.
- Uses mean pixel intensity as movement score.
- Flags stampede risk when score > motion threshold.

### `buffer.py`
- Implements `RollingFrameBuffer` with deque.
- Stores timestamp + frame.
- Max length = `5 minutes * FPS`.
- Old frames auto-evicted when full.

### `lost_item_search.py`
- Encodes uploaded item image with CLIP.
- Runs YOLO over each buffered frame.
- Extracts object crops and embeds via CLIP.
- Computes cosine similarity and returns best match.

### `alerts.py`
- Audio alarm (`alert.wav`) playback.
- Console alert output.
- WhatsApp messaging via Twilio API.
- Reads Twilio credentials from environment variables.

### `dashboard.py`
- Multi-page Streamlit UI:
  - Live feed + metrics + alerts
  - CSV log table
  - Density and alert analytics plots
  - Lost & Found image upload and match result

### `utils.py`
- Event log helpers.
- Timestamp generation.
- RGB/BGR conversion utilities.
- Directory/bootstrap setup.

---

## Installation Guide

### 1) Clone and enter project
```bash
git clone <your-repo-url>
cd AURA
```

### 2) Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

> Note: This project uses `opencv-python-headless` by default so Streamlit previews work in headless/cloud containers (no `libGL.so.1` dependency). If you need local OpenCV GUI windows (`cv2.imshow`) on desktop Linux/Windows, you can switch to `opencv-python`.

### 4) (Optional) Configure Twilio WhatsApp alerts
```bash
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="xxxxxxxxxxxxxxxx"
export TWILIO_WHATSAPP_FROM="whatsapp:+14155238886"
export TWILIO_WHATSAPP_TO="whatsapp:+91xxxxxxxxxx"
```

### 5) Run dashboard
```bash
streamlit run app.py
```

---

## Example Output (Expected)

- **Live Monitoring**
  - Bounding boxes around people.
  - Real-time metrics: People Count, Density, Movement Score, Buffered Frames.
  - Alerts shown on screen when thresholds are crossed.

- **Log Viewer**
  - Event table with columns:
    - `timestamp`
    - `event_type`
    - `description`
    - `people_count`
    - `density`
    - `movement_score`

- **Analytics**
  - Density trend line chart.
  - Alert frequency bar chart.

- **Lost & Found**
  - Uploaded reference image preview.
  - Most similar buffered frame with highlighted object box and similarity score.

---

## Deployment Instructions

### Local Production Mode
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker (recommended for demo portability)
1. Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

2. Build and run:
```bash
docker build -t aura-app .
docker run -p 8501:8501 aura-app
```

### Cloud deployment options
- Streamlit Community Cloud (for UI demos)
- Render / Railway / EC2 / Azure VM
- Kubernetes with horizontal scaling for camera pipelines

---

## Notes for Final Year Project Demonstration

- Keep a prerecorded crowd video for reproducible demo.
- Tune density and movement thresholds live to show alert behavior.
- Demonstrate event logging and analytics after multiple simulated incidents.
- Showcase Lost & Found search with an object image (e.g., bag/bottle).
- Optionally use GPU for higher FPS and better responsiveness.
