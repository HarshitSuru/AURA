"""Streamlit dashboard for AURA system."""

from __future__ import annotations

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from aura_project.alerts import AlertManager
from aura_project.bottleneck_analysis import BottleneckAnalyzer
from aura_project.buffer import RollingFrameBuffer
from aura_project.detection import CrowdDetector
from aura_project.lost_item_search import LostItemSearcher
from aura_project.motion_analysis import MotionAnalyzer
from aura_project.utils import append_event, ensure_directories, load_events, timestamp_now, to_rgb


FPS = 6
FRAME_SIZE = (640, 480)


def init_state() -> None:
    ensure_directories()
    if "buffer" not in st.session_state:
        st.session_state.buffer = RollingFrameBuffer(max_minutes=5, fps=FPS)
    if "detector" not in st.session_state:
        st.session_state.detector = CrowdDetector(model_path="yolov8n.pt")
    if "motion" not in st.session_state:
        st.session_state.motion = MotionAnalyzer(high_movement_threshold=14.0, stampede_threshold=22.0)
    if "bottleneck" not in st.session_state:
        st.session_state.bottleneck = BottleneckAnalyzer(concentration_threshold=0.48)
    if "alerts" not in st.session_state:
        st.session_state.alerts = AlertManager(
            default_alarm_file="alert.wav",
            sound_map={"high_density": "alert.wav", "high_movement": "alert.wav", "stampede": "alert.wav", "bottleneck": "alert.wav"},
        )
    if "density_history" not in st.session_state:
        st.session_state.density_history = []


def log_event(event_type: str, description: str, people_count: int, density: float, movement_score: float) -> None:
    append_event(
        {
            "timestamp": timestamp_now(),
            "event_type": event_type,
            "description": description,
            "people_count": people_count,
            "density": density,
            "movement_score": movement_score,
        }
    )


def render_live_monitoring() -> None:
    st.title("AURA – Live Monitoring")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        source = st.text_input("Camera source (0 for webcam / URL for CCTV)", "0")
    with c2:
        density_threshold = st.slider("Density Threshold", min_value=0.00001, max_value=0.0005, value=0.00008, step=0.00001)
    with c3:
        high_movement_threshold = st.slider("High Movement Threshold", min_value=5.0, max_value=60.0, value=14.0, step=1.0)
    with c4:
        stampede_threshold = st.slider("Stampede Threshold", min_value=5.0, max_value=60.0, value=22.0, step=1.0)
    with c5:
        bottleneck_threshold = st.slider("Bottleneck Threshold", min_value=0.20, max_value=0.90, value=0.48, step=0.01)

    run = st.toggle("Start Monitoring", value=False)
    frame_slot = st.empty()

    detector: CrowdDetector = st.session_state.detector
    motion: MotionAnalyzer = st.session_state.motion
    bottleneck: BottleneckAnalyzer = st.session_state.bottleneck
    alerts: AlertManager = st.session_state.alerts
    frame_buffer: RollingFrameBuffer = st.session_state.buffer

    detector.density_threshold = density_threshold
    motion.high_movement_threshold = high_movement_threshold
    motion.stampede_threshold = max(stampede_threshold, high_movement_threshold)
    bottleneck.concentration_threshold = bottleneck_threshold

    metrics = st.columns(5)
    status_slot = st.empty()

    if not run:
        st.info("Enable start monitoring to begin video analytics.")
        return

    cap_source = 0 if source.strip() == "0" else source.strip()
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        st.error("Could not open camera/CCTV source.")
        return

    twilio_cfg = alerts.from_env()

    while run:
        start = time.time()
        ok, frame = cap.read()
        if not ok:
            status_slot.warning("Frame capture failed; stopping stream.")
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        ts = timestamp_now()

        crowd = detector.detect(frame)
        motion_res = motion.analyze(frame)
        bottleneck_res = bottleneck.analyze(crowd.detections, FRAME_SIZE[0])

        frame_buffer.add(ts, crowd.annotated_frame)
        st.session_state.density_history.append({"timestamp": ts, "density": crowd.density})
        if len(st.session_state.density_history) > 2000:
            st.session_state.density_history = st.session_state.density_history[-2000:]

        metrics[0].metric("People Count", crowd.people_count)
        metrics[1].metric("Density", f"{crowd.density:.6f}")
        metrics[2].metric("Movement Score", f"{motion_res.movement_score:.2f}")
        metrics[3].metric("Bottleneck Score", f"{bottleneck_res.concentration_score:.2f}")
        metrics[4].metric("Buffered Frames", len(frame_buffer))

        alert_msgs = []
        if crowd.is_high_density:
            msg = f"HIGH DENSITY ALERT: crowd density {crowd.density:.6f}"
            alert_msgs.append(msg)
            alerts.play_alarm("high_density")
            log_event("HIGH_DENSITY", msg, crowd.people_count, crowd.density, motion_res.movement_score)

        if motion_res.is_high_movement:
            msg = f"HIGH MOVEMENT ALERT: movement score {motion_res.movement_score:.2f}"
            alert_msgs.append(msg)
            alerts.play_alarm("high_movement")
            log_event("HIGH_MOVEMENT", msg, crowd.people_count, crowd.density, motion_res.movement_score)

        if motion_res.is_stampede_risk:
            msg = f"STAMPEDE RISK ALERT: movement score {motion_res.movement_score:.2f}"
            alert_msgs.append(msg)
            alerts.play_alarm("stampede")
            log_event("STAMPEDE_RISK", msg, crowd.people_count, crowd.density, motion_res.movement_score)

        if bottleneck_res.is_bottleneck:
            zone_l, zone_r = bottleneck_res.hot_zone
            msg = f"BOTTLENECK ALERT: concentration {bottleneck_res.concentration_score:.2f} in zone x={zone_l}-{zone_r}"
            alert_msgs.append(msg)
            alerts.play_alarm("bottleneck")
            log_event("BOTTLENECK", msg, crowd.people_count, crowd.density, motion_res.movement_score)

        if alert_msgs:
            alert_block = " | ".join(alert_msgs)
            status_slot.error(alert_block)
            alerts.console_alert(alert_block)
            if twilio_cfg:
                try:
                    alerts.send_whatsapp(alert_block, twilio_cfg)
                except Exception as exc:
                    status_slot.warning(f"Twilio send failed: {exc}")
        else:
            status_slot.success("System normal")

        frame_slot.image(to_rgb(crowd.annotated_frame), channels="RGB", use_container_width=True)

        elapsed = time.time() - start
        sleep_for = max(0.0, (1.0 / FPS) - elapsed)
        time.sleep(sleep_for)
        run = st.session_state.get("Start Monitoring", True)

    cap.release()


def render_log_viewer() -> None:
    st.title("AURA – Event Log Viewer")
    df = load_events()
    if df.empty:
        st.info("No events logged yet.")
        return
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)


def render_analytics() -> None:
    st.title("AURA – Analytics")
    logs = load_events()
    hist = pd.DataFrame(st.session_state.density_history)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Crowd Density Over Time")
        fig, ax = plt.subplots(figsize=(8, 3))
        if not hist.empty:
            ax.plot(hist["timestamp"], hist["density"], color="royalblue")
            ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Density")
        ax.set_xlabel("Time")
        st.pyplot(fig)

    with col2:
        st.subheader("Alert Frequency")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        if not logs.empty:
            counts = logs["event_type"].value_counts()
            ax2.bar(counts.index, counts.values, color=["crimson", "darkorange"])
        ax2.set_ylabel("Count")
        st.pyplot(fig2)


def render_lost_and_found() -> None:
    st.title("AURA – Lost & Found Search")
    st.caption("Upload an image of the lost item, then search within last 5 minutes of buffered frames.")

    uploaded = st.file_uploader("Upload lost item image", type=["jpg", "jpeg", "png"])
    similarity = st.slider("Similarity Threshold", 0.1, 0.9, 0.28, 0.01)

    if uploaded is None:
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    item = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if item is None:
        st.error("Unable to decode uploaded image.")
        return

    st.image(to_rgb(item), caption="Uploaded item", use_container_width=False)

    if st.button("Run Lost Item Search"):
        with st.spinner("Running CLIP similarity search on buffered frames..."):
            searcher = LostItemSearcher(yolo_model_path="yolov8n.pt")
            match = searcher.search(item, st.session_state.buffer, similarity_threshold=similarity)

        if match:
            st.success(f"Match found at {match.timestamp} with similarity {match.similarity:.3f}")
            st.image(to_rgb(match.annotated_frame), caption="Best match frame", use_container_width=True)
            log_event("LOST_ITEM_MATCH", f"Lost item matched with similarity {match.similarity:.3f}", 0, 0.0, 0.0)
        else:
            st.warning("No confident match found in current frame buffer.")


def run_dashboard() -> None:
    st.set_page_config(page_title="AURA", layout="wide")
    init_state()

    st.sidebar.title("AURA Control Center")
    page = st.sidebar.radio("Select Module", ["LIVE MONITORING", "LOG VIEWER", "ANALYTICS", "LOST & FOUND"])

    if page == "LIVE MONITORING":
        render_live_monitoring()
    elif page == "LOG VIEWER":
        render_log_viewer()
    elif page == "ANALYTICS":
        render_analytics()
    else:
        render_lost_and_found()


if __name__ == "__main__":
    run_dashboard()
