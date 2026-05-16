from pathlib import Path
from typing import Any

import torch
from loguru import logger
from ultralytics import YOLO


class VisionTrainer:
    """
    A class responsible for initializing and training YOLO vision models.
    """

    def __init__(
        self, model_variant: str = "yolov8n.pt", project_root: str = "."
    ) -> None:
        """
        Initializes the VisionTrainer, sets up directory structures, and loads the model.

        Args:
            model_variant (str): The YOLO model weights to load. Defaults to "yolov8n.pt".
            project_root (str): The root path of the project directory. Defaults to ".".

        Returns:
            None
        """
        try:
            # Clear the GPU cache to ensure maximum VRAM availability
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache successfully.")

            self.root: Path = Path(project_root)
            self.checkpoint_dir: Path = self.root / "models" / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            self.model: YOLO = YOLO(model_variant)
            logger.info(f"Model {model_variant} successfully initialized.")

        except Exception as e:
            logger.error(f"Failed to initialize VisionTrainer: {e}")
            raise

    def train_custom_person_detector(
        self, data_yaml: str, epochs: int = 25, imgsz: int = 640
    ) -> Any:
        """
        Executes the optimized training loop for a custom person detection dataset.

        Args:
            data_yaml (str): Path to the YAML file configuring the dataset.
            epochs (int): The number of epochs to train the model. Defaults to 25.
            imgsz (int): The image size to use for training. Defaults to 640.

        Returns:
            Any: The results object returned by the YOLO training engine.
        """
        try:
            logger.info(f"Starting {epochs}-epoch optimized run...")
            results: Any = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,        # UPGRADED: 800px provides better clarity for small people
                project=str(self.checkpoint_dir.parent),
                name="person_detector_v2",
                exist_ok=True,
                batch=8,           
                optimizer="auto",   
                cos_lr=True,        
                freeze=10,          # UPGRADED: Freezes ONLY the Backbone (Layers 0-9). Lets the Neck learn!
                device=0,
		        cache=False,
                patience=15,
                workers=2,          # STABILITY: Keeps Windows from crashing
                amp=False,          # STABILITY: Prevents RTX 40-series illegal instruction faults
                # # --- CROWD DETECTION HYPERPARAMETERS ---
                # mosaic=0.0,         # Prevents shrinking 4 images into 1, keeping people visible
                # mixup=0.0,          
                # scale=0.1           
            )

            logger.success("Success! Results saved to models/person_detector_v2/")
            return results

        except Exception as e:
            logger.error(f"Training failed during execution: {e}")
            raise
