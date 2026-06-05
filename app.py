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
    Centralized inference and annotation logic to prevent code duplication
    between Live Camera and Offline Video modes.
    """
    start_inference = time.time()

    # YOLO Inference + BoT-SORT
    results = model.track(
        source=frame,
        conf=conf,
        iou=iou,
        persist=True,
        tracker="custom_botsort.yaml",
        verbose=False,
        imgsz=640,
        half=True
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
    """Main Streamlit application loop and UI renderer."""
    setup_page()

    # --- 1. SIDEBAR CONTROLS ---
    st.sidebar.header("Pipeline Configuration")
    
    # Input Mode Selection
    input_mode = st.sidebar.radio("Data Source", ["Live Camera", "Upload Video"])
    
    video_path: Any = None
    selected_stream_name: str = "Custom_Upload"

    if input_mode == "Live Camera":
        available_streams: Dict[str, Any] = {
            "Front_Desk (Webcam)": 0
        }
        selected_stream_name = st.sidebar.selectbox("Active Camera", list(available_streams.keys()))
        video_path = available_streams[selected_stream_name]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an MP4 or AVI", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            # Create a temporary file on the local OS to feed to OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            
            # --- WINDOWS FIX: Release the write-lock so OpenCV can read it and OS can delete it later ---
            tfile.close() 
            
            video_path = tfile.name

    # Threshold Adjustments
    st.sidebar.subheader("Model Hyperparameters")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
    iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.1, 1.0, 0.50, 0.05)
    
    start_button = st.sidebar.button("Initialize Pipeline", type="primary")
    stop_button = st.sidebar.button("Halt Pipeline")

    # --- 2. MAIN LAYOUT: PERFORMANCE METRICS ---
    st.markdown("### System Telemetry")
    col1, col2, col3 = st.columns(3)
    fps_metric = col1.empty()
    latency_metric = col2.empty()
    count_metric = col3.empty()

    st.markdown("### Vision Feed")
    video_placeholder = st.empty()

    # --- 3. EXECUTION LOOP ---
    if start_button:
        if input_mode == "Upload Video" and video_path is None:
            st.error("🚨 Please upload a video file before initializing the pipeline.")
            st.stop()

        try:
            logger.info(f"Initializing {input_mode} pipeline...")
            model = YOLO("models/checkpoints/best_quantized.onnx", task='detect')
            
            counter = ROICounter()

            prev_time = time.time()

            # --- ROUTE A: LIVE CAMERA LOOP ---
            if input_mode == "Live Camera":
                active_source = {selected_stream_name: video_path}
                stream_manager = MultiStreamManager(sources=active_source)
                stream_manager.start()
                
                retry_count = 0
                while not stop_button:
                    frames = stream_manager.get_frames()
                    frame = frames.get(selected_stream_name)

                    if frame is None:
                        retry_count += 1
                        if retry_count > 100:
                            st.error("🚨 Camera feed lost. Ensure no other app is using the webcam.")
                            break
                        time.sleep(0.01)
                        continue
                    
                    retry_count = 0
                    
                    # Run central inference
                    annotated_frame, live_count, inference_time_ms = _process_frame(
                        frame, model, counter, conf_threshold, iou_threshold
                    )

                    # Update UI
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", width="stretch")

                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time

                    fps_metric.metric("Pipeline FPS", f"{fps:.1f}")
                    latency_metric.metric("Inference Latency", f"{inference_time_ms:.1f} ms")
                    count_metric.metric("Zone Occupancy", str(live_count))
                    
                stream_manager.stop()

            # --- ROUTE B: OFFLINE VIDEO LOOP ---
            else:
                cap = cv2.VideoCapture(video_path)
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.success("✅ Video processing complete!")
                        break
                    
                    # Run central inference
                    annotated_frame, live_count, inference_time_ms = _process_frame(
                        frame, model, counter, conf_threshold, iou_threshold
                    )

                    # Update UI
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", width="stretch")

                    curr_time = time.time()
                    fps = 1 / (max(curr_time - prev_time, 0.001)) # Prevent divide by zero on fast frames
                    prev_time = curr_time

                    fps_metric.metric("Pipeline FPS", f"{fps:.1f}")
                    latency_metric.metric("Inference Latency", f"{inference_time_ms:.1f} ms")
                    count_metric.metric("Zone Occupancy", str(live_count))

                cap.release()

        except Exception as e:
            logger.error(f"UI Loop Exception: {e}")
            st.error(f"Pipeline interrupted: {e}")
        finally:
            logger.info("Pipeline offline.")
            # Clean up the temporary OS file if a video was uploaded
            if input_mode == "Upload Video" and video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.info("Temporary video file purged from system.")
                except Exception as e:
                    logger.warning(f"Could not delete temp video file: {e}")

if __name__ == "__main__":
    main()