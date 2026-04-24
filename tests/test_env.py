import torch
import cv2
import supervision as sv
from ultralytics import YOLO


def test_imports():
    """Verify all core CV libraries are accessible."""
    assert torch.__version__ is not None
    assert cv2.__version__ is not None
    assert sv.__version__ is not None
    print("✅ All core libraries imported successfully.")


def test_hardware_check():
    """Log whether CUDA or CPU is being used in the environment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ℹ️ Testing environment is running on: {device}")
    # We don't assert CUDA here because GitHub Actions runners are CPU-only by default
