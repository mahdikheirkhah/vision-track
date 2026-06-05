import numpy as np
import supervision as sv
from loguru import logger
from typing import Tuple

class ROICounter:
    """
    Tracks and counts unique people across the ENTIRE screen.
    Handles visual annotations (Boxes, Labels, Trails). No ROI polygons.
    """

    def __init__(self):
        try:
            # Individual Tracking Visuals (Boxes, IDs, and Trails)
            # FIXED: Updated to modern sv.BoxAnnotator for Supervision v0.27+
            self.box_annotator = sv.BoxAnnotator(thickness=2)
            self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
            self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
            
            # Persistent Memory for Unique IDs
            self.unique_tracked_ids = set()

            logger.info("ROICounter initialized for Full-Screen Tracking.")
            
        except Exception as e:
            logger.error(f"Failed to initialize ROICounter: {e}")
            raise

    def process_and_annotate(
        self, 
        frame: np.ndarray, 
        detections: sv.Detections
    ) -> Tuple[np.ndarray, int]:
        
        try:
            # 1. Update Persistent Memory with ALL detected IDs on screen
            if detections.tracker_id is not None:
                for t_id in detections.tracker_id:
                    self.unique_tracked_ids.add(t_id)

            cumulative_count = len(self.unique_tracked_ids)
            
            # 2. Generate Tracking Labels
            labels = []
            if detections.tracker_id is not None:
                for tracker_id, conf in zip(detections.tracker_id, detections.confidence):
                    labels.append(f"ID:{tracker_id} {conf:.2f}")
            else:
                for conf in detections.confidence:
                    labels.append(f"{conf:.2f}")

            # 3. Apply Visual Layers
            annotated_frame = frame.copy()

            # Layer A: Trails
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            
            # Layer B: Bounding Boxes (Rectangles around people)
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            
            # Layer C: Text Labels (ID and Confidence)
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            
            return annotated_frame, cumulative_count

        except Exception as e:
            logger.warning(f"Failed to process annotations. Error: {e}")
            return frame, len(self.unique_tracked_ids)