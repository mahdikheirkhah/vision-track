import os
import json
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any, List, Set
from pycocotools.coco import COCO
import os
import random
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

    def generate_labels(
        self,
        background_ratio: float = 0.10,
        max_person_images: int = None
        ) -> None:
        """
        Generates YOLO format labels, including a specific ratio of background 
        images (images with no people) to reduce False Positives during training.

        Args:
            background_ratio (float): The ratio of background images to person images. 
                                      0.10 means 10% background images.
            max_person_images (int, optional): The maximum number of person images to include. 
                                               Useful for creating smaller datasets for faster training.
        """
        try:
            # 1. Get ALL image IDs in the COCO dataset
            all_img_ids: Set[int] = set(self.coco.getImgIds())
            
            # 2. Get EVERY single image ID that contains a person
            all_person_img_ids: Set[int] = set(self.coco.getImgIds(catIds=[self.person_cat_id]))
            
            # 3. Create our working list of person images to process
            person_img_ids_to_process: List[int] = list(all_person_img_ids)
            
            # --- NEW CAPABILITY: Subsample person images if requested ---
            if max_person_images is not None and max_person_images < len(person_img_ids_to_process):
                logger.info(f"Subsampling dataset: Limiting from {len(person_img_ids_to_process)} to {max_person_images} person images.")
                person_img_ids_to_process = random.sample(person_img_ids_to_process, max_person_images)
            
            # 4. Safely calculate background images
            # We MUST subtract `all_person_img_ids` (not the subsample) so we don't 
            # accidentally pick a left-out person image as an empty background!
            safe_background_img_ids: List[int] = list(all_img_ids - all_person_img_ids)
            
            # 5. Calculate how many background images we need based on our SUBSAMPLED person count
            num_background_to_keep: int = int(len(person_img_ids_to_process) * background_ratio)
            
            # Randomly sample the background images
            sampled_background_ids: List[int] = random.sample(
                safe_background_img_ids, 
                min(num_background_to_keep, len(safe_background_img_ids))
            )
            
            # Combine them for the final processing list
            final_img_ids: List[int] = person_img_ids_to_process + sampled_background_ids
            
            logger.info(f"Processing {len(person_img_ids_to_process)} images with people.")
            logger.info(f"Processing {len(sampled_background_ids)} background images (Empty labels).")
            
            for img_id in final_img_ids:
                self._process_single_image(img_id)
                
            logger.success("Label generation complete!")

        except Exception as e:
            logger.error(f"Failed to generate labels: {e}")
            raise

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
                bbox = ann['bbox']
                # Convert to YOLO: [x_center, y_center, width, height] normalized 0-1
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                y_width = bbox[2] / w
                y_height = bbox[3] / h
                
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {y_width:.6f} {y_height:.6f}")

            # Write to file. 
            # Note: If `yolo_lines` is empty (a background image), this safely creates an empty .txt file!
            txt_filename = file_name.replace('.jpg', '.txt')
            output_path = self.output_dir / txt_filename
            
            with open(output_path, 'w') as f:
                f.write("\n".join(yolo_lines))
                
        except Exception as e:
            logger.error(f"Failed to process image {img_id}: {e}")
            raise