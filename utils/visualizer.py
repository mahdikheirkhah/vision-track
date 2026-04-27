import cv2
import supervision as sv
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt

class YOLOVisualizer:
    """
    Visualizes YOLO-formatted labels to verify COCO conversion accuracy.
    Uses the modern sv.DetectionDataset approach for robust verification.
    """
    def __init__(self, dataset_path: str, images_dir: str, labels_dir: str):
        """
        Args:
            dataset_path: Path to your data.yaml
            images_dir: Directory of images (e.g., data/coco/images/val2017)
            labels_dir: Directory of labels (e.g., data/coco/labels/val2017)
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

    def preview(self, num_samples: int = 3) -> None:
        """Loads the dataset and plots samples to verify conversion."""
        try:
            dataset = sv.DetectionDataset.from_yolo(
                images_directory_path=str(self.images_dir),
                annotations_directory_path=str(self.labels_dir),
                data_yaml_path=str(self.dataset_path)
            )
            
            logger.info(f"Dataset loaded. Total images: {len(dataset)}")

            box_annotator = sv.BoxAnnotator()
            
            # FIXED: Unpacking 3 values: image_path, image, and detections
            for i, (image_path, image, detections) in enumerate(dataset):
                if i >= num_samples:
                    break

                # Convert BGR to RGB for plotting
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Annotate the frame
                annotated_frame = box_annotator.annotate(
                    scene=image_rgb.copy(), 
                    detections=detections
                )
                
                plt.figure(figsize=(10, 10))
                plt.imshow(annotated_frame)
                plt.title(f"Verified: {Path(image_path).name}")
                plt.axis("off")
                plt.show()

            logger.success(f"Visual audit of {num_samples} samples complete.")

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise