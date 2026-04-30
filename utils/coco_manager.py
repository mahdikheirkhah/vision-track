import os
import json
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any
from pycocotools.coco import COCO

class COCOManager:
    """
    Handles conversion from COCO JSON format to YOLOv8 normalized text format.
    Filters specifically for the 'person' category (Category ID: 1).
    """

    def __init__(self, annotation_path: str, image_dir: str, output_dir: str):
        """
        Args:
            annotation_path: Path to instances_val2017.json
            image_dir: Directory where the .jpg files live
            output_dir: Where to save the generated YOLO .txt labels
        """
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.coco = COCO(str(self.annotation_path))
            # COCO standard: 'person' is category ID 1
            self.person_cat_id = self.coco.getCatIds(catNms=['person'])[0]
            logger.info(f"COCO initialized. 'person' Category ID: {self.person_cat_id}")
        except Exception as e:
            logger.error(f"Failed to initialize COCO API: {e}")
            raise

    def convert_to_yolo(self) -> None:
        """Iterates through images containing people and generates YOLO labels."""
        try:
            img_ids = self.coco.getImgIds(catIds=[self.person_cat_id])
            logger.info(f"Found {len(img_ids)} images containing people.")

            for img_id in img_ids:
                self._process_single_image(img_id)
                
            logger.success(f"Conversion complete. Labels saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error during conversion: {e}")

    def _process_single_image(self, img_id: int) -> None:
        """Converts annotations for one image to a YOLO .txt file."""
        try:
            img_info = self.coco.loadImgs(img_id)[0]
            w, h = img_info['width'], img_info['height']
            file_name = img_info['file_name']
            
            # Find all person annotations for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_cat_id])
            anns = self.coco.loadAnns(ann_ids)

            yolo_lines = []
            for ann in anns:
                # COCO format: [x_min, y_min, width, height]
                bbox = ann['bbox']
                
                # Convert to YOLO: [x_center, y_center, width, height] normalized 0-1
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                y_width = bbox[2] / w
                y_height = bbox[3] / h
                
                # '0' is the class index for YOLO (since we only have 1 class: person)
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {y_width:.6f} {y_height:.6f}")

            # Write to file
            txt_name = Path(file_name).stem + ".txt"
            with open(self.output_dir / txt_name, "w") as f:
                f.write("\n".join(yolo_lines))
                
        except Exception as e:
            logger.warning(f"Failed to process image {img_id}: {e}")