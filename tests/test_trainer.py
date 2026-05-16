import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.trainer import VisionTrainer


def test_vision_trainer_initialization_success(tmp_path: Path) -> None:
    """
    Tests that the VisionTrainer initializes correctly, clears CUDA cache, 
    and creates the necessary checkpoint directories.
    """
    with patch("trainer.torch.cuda.is_available", return_value=True), \
         patch("trainer.torch.cuda.empty_cache") as mock_empty_cache, \
         patch("trainer.YOLO") as mock_yolo:
        
        trainer = VisionTrainer(model_variant="yolov8n.pt", project_root=str(tmp_path))
        
        # Verify CUDA cache clearing was triggered
        mock_empty_cache.assert_called_once()
        # Verify YOLO was instantiated with the correct weights
        mock_yolo.assert_called_once_with("yolov8n.pt")
        # Verify directory structure was created
        assert trainer.checkpoint_dir.exists()


def test_vision_trainer_initialization_failure(tmp_path: Path) -> None:
    """
    Tests that the __init__ exception block catches and logs errors properly.
    """
    with patch("trainer.YOLO", side_effect=RuntimeError("Mocked YOLO Init Error")):
        with pytest.raises(RuntimeError, match="Mocked YOLO Init Error"):
            VisionTrainer(model_variant="invalid_model.pt", project_root=str(tmp_path))


def test_train_custom_person_detector_success(tmp_path: Path) -> None:
    """
    Tests that the training method executes successfully and returns the expected results.
    """
    with patch("trainer.YOLO"):
        trainer = VisionTrainer(project_root=str(tmp_path))
        # Mock the train method to bypass actual GPU training
        trainer.model.train = MagicMock(return_value={"map50": 0.95})
        
        results = trainer.train_custom_person_detector(data_yaml="dummy.yaml", epochs=5)
        
        assert results == {"map50": 0.95}
        trainer.model.train.assert_called_once()


def test_train_custom_person_detector_failure(tmp_path: Path) -> None:
    """
    Tests that the training exception block catches and re-raises runtime errors.
    """
    with patch("trainer.YOLO"):
        trainer = VisionTrainer(project_root=str(tmp_path))
        # Force the train method to fail
        trainer.model.train = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            trainer.train_custom_person_detector(data_yaml="dummy.yaml", epochs=5)