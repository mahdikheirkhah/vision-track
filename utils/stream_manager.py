import cv2
import threading
import time
from loguru import logger
from typing import Dict, List, Optional, Tuple
import numpy as np
import sys

class MultiStreamManager:
    """
    Orchestrates multiple concurrent OpenCV video streams using background threads.
    Guarantees zero-latency frame fetching by actively clearing hardware buffers.
    """

    def __init__(self, sources: Dict[str, str]):
        """
        Args:
            sources (Dict[str, str]): A dictionary mapping stream names to their URLs/paths.
                                      e.g., {"Cam_1": "rtsp://...", "Cam_2": 0}
        """
        self.sources = sources
        self.captures: Dict[str, cv2.VideoCapture] = {}
        self.latest_frames: Dict[str, Optional[np.ndarray]] = {name: None for name in sources}
        
        # Thread control flags
        self.is_running = False
        self.threads: List[threading.Thread] = []

    def start(self) -> None:
        """Initializes hardware connections and spawns background reader threads."""
        self.is_running = True
        
        for name, source in self.sources.items():
            try:
                # --- WINDOWS FIX: Force DirectShow backend for integer webcams ---
                if isinstance(source, int) and sys.platform == "win32":
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(source)
                # -----------------------------------------------------------------

                if not cap.isOpened():
                    logger.error(f"Failed to open hardware stream: {name} at {source}")
                    continue
                
                self.captures[name] = cap
                
                thread = threading.Thread(target=self._update_loop, args=(name,), daemon=True)
                thread.start()
                self.threads.append(thread)
                logger.success(f"Stream thread activated for: {name}")
                
            except Exception as e:
                logger.error(f"Critical failure starting stream {name}: {e}")

    def _update_loop(self, name: str) -> None:
        """
        The background thread payload. Constantly grabs the newest frame to 
        prevent OpenCV buffer lag.
        """
        cap = self.captures[name]
        while self.is_running:
            ret, frame = cap.read()
            if ret:
                # Atomically update the dictionary with the newest frame
                self.latest_frames[name] = frame
            else:
                logger.warning(f"Stream {name} dropped a frame or disconnected.")
                time.sleep(0.1)  # Prevent CPU thrashing on disconnect

    def get_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Pulls the most recent frames from all active cameras instantly.
        
        Returns:
            Dict[str, np.ndarray]: Mapping of camera names to their current BGR frames.
        """
        return self.latest_frames

    def stop(self) -> None:
        """Safely shuts down threads and releases hardware bindings."""
        logger.info("Initiating MultiStreamManager shutdown sequence...")
        self.is_running = False
        
        for thread in self.threads:
            thread.join(timeout=1.0)
            
        for name, cap in self.captures.items():
            cap.release()
            logger.info(f"Released hardware bindings for {name}")