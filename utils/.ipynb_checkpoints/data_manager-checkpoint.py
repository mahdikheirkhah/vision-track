import os
import yaml
from pathlib import Path
from loguru import logger
from typing import Dict, List, Tuple, Any

class DataOrganizer:
    """
    Manages dataset paths and file pairs for VisionTrack.
    Follows strict OOP and Loguru standards.
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = Path(yaml_path)
        self.root = self.yaml_path.parent
        self.config = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        """Loads the Roboflow dataset configuration."""
        try:
            with open(self.yaml_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load data.yaml: {e}")
            raise

    def get_image_label_pairs(self, split: str = "train") -> List[Tuple[Path, Path]]:
        """
        Pairs images with their corresponding YOLO labels.
        
        Args:
            split (str): 'train', 'val', or 'test'.
        """
        try:
            # Resolve relative paths from YAML (e.g., '../train/images')
            img_dir = (self.root / self.config[split]).resolve()
            label_dir = img_dir.parent / "labels"
            logger.info(f"image directory: {img_dir}")
            logger.info(f"label directory: {label_dir}")
            pairs = []
            for img_path in img_dir.glob("*.jpg"):
                label_path = label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    pairs.append((img_path, label_path))
            
            logger.info(f"Found {len(pairs)} valid pairs in {split} split.")
            return pairs
        except Exception as e:
            logger.error(f"Error pairing files: {e}")
            return []