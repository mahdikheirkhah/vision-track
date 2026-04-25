import os
import shutil
from pathlib import Path
from loguru import logger
from typing import Union, List

class DataOrganizer:
    """
    Handles the structured organization of raw videos and COCO datasets
    into a YOLOv8-compatible directory tree.
    """

    def __init__(self, root_dir: str = "data"):
        """
        Initializes the organizer with a root data directory.
        
        Args:
            root_dir (str): The base directory for data storage.
        """
        self.root = Path(root_dir)
        self.raw_path = self.root / "raw_videos"
        self.coco_path = self.root / "coco_dataset"
        self._init_structure()

    def _init_structure(self) -> None:
        """Creates the initial directory structure if it doesn't exist."""
        try:
            for path in [self.raw_path, self.coco_path]:
                path.mkdir(parents=True, exist_ok=True)
            logger.success("Data directory structure verified.")
        except Exception as e:
            logger.error(f"Failed to initialize directories: {e}")
            raise

    def organize_raw_video(self, source_path: str) -> bool:
        """
        Moves a raw video into the managed raw_videos folder.
        
        Args:
            source_path (str): Current path of the video file.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            src = Path(source_path)
            if not src.exists():
                logger.warning(f"Source file not found: {source_path}")
                return False
            
            dest = self.raw_path / src.name
            shutil.move(str(src), str(dest))
            logger.info(f"Organized video: {src.name} -> {self.raw_path}")
            return True
        except Exception as e:
            logger.error(f"Error organizing video {source_path}: {e}")
            return False