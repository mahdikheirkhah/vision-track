import cv2
import numpy as np
from loguru import logger
from typing import Tuple, Optional

class Preprocessor:
    """
    Implements normalization and resizing logic for VisionTrack frames
    prior to model inference.
    """

    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Resizes frame to target dimensions using inter-linear interpolation.
        
        Args:
            frame (np.ndarray): The input image/frame.
            target_size (Tuple[int, int]): Width and Height.
            
        Returns:
            np.ndarray: The resized frame.
        """
        try:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            logger.error(f"Resizing failed: {e}")
            return frame

    @staticmethod
    def normalize(frame: np.ndarray) -> np.ndarray:
        """
        Normalizes pixel values from [0, 255] to [0.0, 1.0].
        
        Args:
            frame (np.ndarray): The input image/frame.
            
        Returns:
            np.ndarray: Normalized float32 array.
        """
        try:
            return frame.astype(np.float32) / 255.0
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return frame