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
        
    def get_split_paths(self, split: str):
        raw_path = self.config.get(split)
        if not raw_path:
            raise ValueError(f"Split '{split}' not found in yaml config.")

        # FIX: Robustly handle Roboflow's '../' quirk
        # If it's a relative path starting with .., we anchor it to the yaml folder
        if raw_path.startswith(".."):
            # .parts[1:] removes the '..' component effectively
            clean_path = Path(*Path(raw_path).parts[1:])
            img_dir = (self.yaml_path / clean_path).resolve()
        else:
            # Use 'path' key if it exists, otherwise assume relative to yaml
            base_path = self.config.get('path', self.yaml_path)
            img_dir = (Path(base_path) / raw_path).resolve()

        # In your structure, labels are siblings to images: train/images -> train/labels
        label_dir = img_dir.parent / "labels"
        return img_dir, label_dir
    
    def get_image_label_pairs(self, split: str = "train") -> List[Tuple[Path, Path]]:
        """
        Pairs images with Roboflow-formatted labels using pattern matching.
        """
        try:
            img_dir, label_dir = self.get_split_paths(split)
            logger.info(f"Searching {split} split in: {img_dir}")

            pairs = []
            # Roboflow images often have .jpg extension
            images = list(img_dir.glob("*.jpg"))
            
            for img_path in images:
                # Roboflow format: [image_stem]_jpg.rf.[hash].txt
                # We search for any .txt file starting with the image stem
                search_pattern = f"{img_path.stem}*.txt"
                matching_labels = list(label_dir.glob(search_pattern))
                
                if matching_labels:
                    # Take the first match (usually only one exists)
                    pairs.append((img_path, matching_labels[0]))
                else:
                    logger.warning(f"No label found for {img_path.name} using pattern {search_pattern}")

            logger.success(f"Successfully paired {len(pairs)}/{len(images)} files.")
            return pairs

        except Exception as e:
            logger.error(f"Critical error during file pairing: {e}")
            return []
        
        
        
        