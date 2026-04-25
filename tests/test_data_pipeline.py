import os
import pytest
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
from typing import Generator

# Import our classes - assuming they are in utils/
from utils.data_manager import DataOrganizer
from utils.preprocessing import Preprocessor

class TestDataPipeline:
    """
    Suite of tests for validating Data Management and Preprocessing logic.
    Follows strict OOP and type hinting standards.
    """

    @pytest.fixture
    def temp_root(self, tmp_path: Path) -> Generator[Path, None, None]:
        """Fixture to provide a clean temporary data directory for each test."""
        yield tmp_path

    @pytest.fixture
    def organizer(self, temp_root: Path) -> DataOrganizer:
        """Fixture to initialize the DataOrganizer with a temporary root."""
        return DataOrganizer(root_dir=str(temp_root))

    def test_organizer_directory_creation(self, temp_root: Path, organizer: DataOrganizer) -> None:
        """Verify that the organizer correctly initializes the folder structure."""
        try:
            assert (temp_root / "raw_videos").exists()
            assert (temp_root / "coco_dataset").exists()
            logger.success("Directory structure validation passed.")
        except AssertionError as e:
            logger.error(f"Directory creation failed: {e}")
            raise

    def test_organize_raw_video_success(self, temp_root: Path, organizer: DataOrganizer) -> None:
        """Test moving a valid video file into the managed structure."""
        try:
            # Create a dummy video file
            fake_video = temp_root / "test_clip.mp4"
            fake_video.write_text("dummy content")
            
            result = organizer.organize_raw_video(str(fake_video))
            
            assert result is True
            assert (temp_root / "raw_videos" / "test_clip.mp4").exists()
            assert not fake_video.exists()  # Should have been moved
            logger.info("Video organization success test passed.")
        except Exception as e:
            logger.error(f"Video organization logic failed: {e}")
            raise

    def test_organize_raw_video_missing_file(self, organizer: DataOrganizer) -> None:
        """Verify handling of non-existent source files."""
        result = organizer.organize_raw_video("non_existent_path.mp4")
        assert result is False
        logger.info("Correctly handled missing file error.")

    def test_preprocessor_resize(self) -> None:
        """Validate frame resizing dimensions."""
        try:
            dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            target = (640, 640)
            
            resized = Preprocessor.resize_frame(dummy_frame, target)
            
            assert resized.shape == (640, 640, 3)
            logger.success("Preprocessor resizing validated.")
        except Exception as e:
            logger.error(f"Resizing test failed: {e}")
            raise

    def test_preprocessor_normalization(self) -> None:
        """Verify pixel normalization is within [0.0, 1.0]."""
        try:
            dummy_frame = np.array([[0, 127, 255]], dtype=np.uint8)
            normalized = Preprocessor.normalize(dummy_frame)
            
            assert normalized.max() <= 1.0
            assert normalized.min() >= 0.0
            assert normalized.dtype == np.float32
            logger.success("Preprocessor normalization validated.")
        except Exception as e:
            logger.error(f"Normalization test failed: {e}")
            raise