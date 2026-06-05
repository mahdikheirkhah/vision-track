import streamlit as st
import cv2
import time
import numpy as np
import sys
import tempfile
import os
from loguru import logger
from typing import Dict, Tuple, Any
from ultralytics import YOLO
import supervision as sv

# Custom module imports
from utils.stream_manager import MultiStreamManager
from utils.counting_logic import ROICounter

def setup_page() -> None:
    """Configures the Streamlit page layout and styling."""
    st.set_page_config(
        page_title="VisionTrack UI",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("VisionTrack Analytics Dashboard")

def _process_frame(
    frame: np.ndarray, 
    model: YOLO, 
    counter: ROICounter, 
    conf: float, 
    iou: float
) -> Tuple[np.ndarray, int, float]:
    """
    Centralized inference and annotation logic.
    """
    start_inference = time.time()

    # YOLO Inference + Custom BoT-SORT
    results = model.track(
        source=frame,
        conf=conf,
        iou=iou,
        persist=True,
        tracker="custom_botsort.yaml",
        verbose=False,
        imgsz=640,
        half=True,
        device=0
    )[0]
    
    inference_time_ms = (time.time() - start_inference) * 1000

    # Supervision Analytics
    detections = sv.Detections.from_ultralytics(results)
    annotated_frame, live_count = counter.process_and_annotate(
        frame=frame,
        detections=detections
    )

    return annotated_frame, live_count, inference_time_ms

def main() -> None:
    setup_page()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Pipeline Configuration")
    input_mode = st.sidebar.radio("Data Source", ["Live Camera", "Upload Video"])
    
    video_path: Any = None
    selected_stream_name: str = "Custom_Upload"

    if input_mode == "Live Camera":
        available_streams = {"Front_Desk (Webcam)": 0}
        selected_stream_name = st.sidebar.selectbox("Active Camera", list(available_streams.keys()))
        video_path = available_streams[selected_stream_name]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an MP4 or AVI", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close() 
            video_path = tfile.name

    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
    iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.1, 1.0, 0.50, 0.05)
    start_button = st.sidebar.button("Initialize Pipeline", type="primary")
    stop_button = st.sidebar.button("Halt Pipeline")

    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    fps_metric = col1.empty()
    latency_metric = col2.empty()
    count_metric = col3.empty()
    video_placeholder = st.empty()

    # --- EXECUTION LOOP ---
    if start_button:
        if input_mode == "Upload Video" and video_path is None:
            st.error("🚨 Please upload a video file.")
            st.stop()

        cap = None  # Failsafe variable
        try:
            logger.info(f"Initializing {input_mode} pipeline...")
            model = YOLO("models/checkpoints/best_quantized.onnx", task='detect')
            counter = ROICounter()
            prev_time = time.time()

            # ROUTE A: CAMERA
            if input_mode == "Live Camera":
                stream_manager = MultiStreamManager(sources={selected_stream_name: video_path})
                stream_manager.start()
                while not stop_button:
                    frame = stream_manager.get_frames().get(selected_stream_name)
                    if frame is None:
                        time.sleep(0.01); continue
                    
                    annotated_frame, live_count, inf_ms = _process_frame(frame, model, counter, conf_threshold, iou_threshold)
                    video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                    
                    curr = time.time()
                    fps_metric.metric("Pipeline FPS", f"{1/(curr - prev_time):.1f}")
                    latency_metric.metric("Inference Latency", f"{inf_ms:.1f} ms")
                    count_metric.metric("Total Unique Visitors", str(live_count))
                    prev_time = curr
                stream_manager.stop()

            # ROUTE B: OFFLINE VIDEO
            else:
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    annotated_frame, live_count, inf_ms = _process_frame(frame, model, counter, conf_threshold, iou_threshold)
                    video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                    
                    curr = time.time()
                    fps_metric.metric("Pipeline FPS", f"{1/(max(curr - prev_time, 0.001)):.1f}")
                    latency_metric.metric("Inference Latency", f"{inf_ms:.1f} ms")
                    count_metric.metric("Total Unique Visitors", str(live_count))
                    prev_time = curr

        finally:
            if cap is not None: cap.release()
            if input_mode == "Upload Video" and video_path and os.path.exists(video_path):
                os.remove(video_path)
            logger.info("Pipeline offline.")

if __name__ == "__main__":
    main()