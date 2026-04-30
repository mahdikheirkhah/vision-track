from ultralytics import YOLO
from loguru import logger
from pathlib import Path

class VisionTrainer:
    """
    Handles YOLOv8 training, fine-tuning, and artifact management.
    """
    def __init__(self, model_variant: str = "yolov8n.pt", project_root: str = "."):
        self.root = Path(project_root)
        self.checkpoint_dir = self.root / "models" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the pre-trained weight (nano is best for your first laptop run)
        self.model = YOLO(model_variant)
        logger.info(f"Model {model_variant} initialized.")

    def train_custom_person_detector(self, data_yaml: str, epochs: int = 50, imgsz: int = 640):
        """
        Executes the fine-tuning process.
        """
        try:
            logger.info(f"Starting training on {data_yaml} for {epochs} epochs...")
            
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                project=str(self.checkpoint_dir.parent),
                name="person_detector",
                exist_ok=True,
                # Hyperparameters for transfer learning
                freeze=10,  # Freezes the first 10 layers (the backbone)
                patience=10, # Early stopping if no improvement for 10 epochs
                device=0     # Uses your GPU (change to 'cpu' if no NVIDIA GPU)
            )
            
            logger.success("Training complete. Artifacts saved to models/person_detector/")
            return results
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise