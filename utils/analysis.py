import pandas as pd
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
from loguru import logger
import json
import shutil
from pathlib import Path
from typing import Dict, Any
from ultralytics import YOLO
class AnalysisManager:
    """
    Performs statistical EDA on YOLO datasets.
    Calculates distribution, box density, and aspect ratios.
    """
    def __init__(self, dataset: sv.DetectionDataset):
        self.dataset = dataset
        self.df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        """Flattens dataset into a searchable Pandas DataFrame."""
        data = []
        for img_name, _, detections in self.dataset:
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                data.append({
                    "image": img_name,
                    "class_id": detections.class_id[i],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1)
                })
        return pd.DataFrame(data)

    def plot_distributions(self):
        """Visualizes class frequency and object sizes."""
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Class Distribution
        self.df['class_id'].value_counts().plot(kind='bar', ax=ax[0])
        ax[0].set_title("Class Frequency")
        
        # Object Size Distribution (Crucial for anchor box tuning)
        self.df['area'].hist(bins=50, ax=ax[1])
        ax[1].set_title("Object Area Distribution (Pixels)")
        
        plt.show()
        logger.info(f"Analyzed {len(self.df)} total bounding boxes.")



class DatasetAnalyzer:
    """
    Performs statistical analysis on YOLO datasets to detect bias and imbalance.
    """
    def __init__(self, dataset: sv.DetectionDataset):
        self.dataset = dataset
        self.stats_df = self._extract_stats()

    def _extract_stats(self) -> pd.DataFrame:
        """Flattens the dataset into a DataFrame for easy analysis."""
        records = []
        for img_name, _, detections in self.dataset:
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                records.append({
                    "file": img_name,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area_px": (x2 - x1) * (y2 - y1),
                    "aspect_ratio": (y2 - y1) / (x2 - x1) if (x2-x1) > 0 else 0
                })
        return pd.DataFrame(records)

    def generate_report(self):
        """Visualizes the critical metrics for object detection."""
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Object Density (People per image)
        counts = self.stats_df.groupby('file').size()
        counts.hist(bins=20, ax=ax[0], color='skyblue', edgecolor='black')
        ax[0].set_title(f"Crowd Density (Avg: {counts.mean():.1f} people/img)")
        ax[0].set_xlabel("Number of People")

        # 2. Box Size Distribution
        self.stats_df['area_px'].plot(kind='hist', bins=50, ax=ax[1], logy=True)
        ax[1].set_title("Object Area (Log Scale)")
        ax[1].set_xlabel("Area in Pixels")
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Analysis complete: {len(self.stats_df)} total person-instances found.")


class MetricsEvaluator:
    """
    Evaluates YOLO training results, enforces strict performance thresholds, 
    generates formatted JSON reports, and handles model artifact exportation (ONNX).
    """

    def __init__(
        self, 
        run_dir: str = "models/person_detector_v2", 
        checkpoints_dir: str = "models/checkpoints",
        reports_dir: str = "reports"
    ) -> None:
        """
        Initializes the MetricsEvaluator with required directory paths.

        Args:
            run_dir (str): The directory where the latest YOLO training run was saved.
            checkpoints_dir (str): The target directory required by the audit for weights.
            reports_dir (str): The target directory required by the audit for JSON metrics.
            
        Returns:
            None
        """
        try:
            self.run_dir: Path = Path(run_dir)
            
            self.checkpoints_dir: Path = Path(checkpoints_dir)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            
            self.reports_dir: Path = Path(reports_dir)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to initialize MetricsEvaluator: {e}")
            raise

    def process_and_export_artifacts(self) -> None:
        """
        Moves the best PyTorch model to the checkpoints folder and exports a quantized ONNX version.

        Returns:
            None
        """
        try:
            source_weights: Path = self.run_dir / "weights" / "best.pt"
            target_weights: Path = self.checkpoints_dir / "best.pt"
            onnx_weights: Path = self.checkpoints_dir / "best_quantized.onnx"

            if not source_weights.exists():
                raise FileNotFoundError(f"Trained weights not found at {source_weights}")

            # 1. Move the best PyTorch model to the audit-required path
            shutil.copy2(source_weights, target_weights)
            logger.info(f"Successfully copied PyTorch weights to {target_weights}")

            # 2. Load model and export to ONNX (Quantized/Half-precision for speedup)
            logger.info("Starting ONNX export and quantization. This may take a moment...")
            model = YOLO(str(target_weights))
            export_path = model.export(format="onnx", half=True, simplify=True)
            
            # 3. Rename output to match audit specifications
            exported_file: Path = Path(export_path)
            exported_file.rename(onnx_weights)
            logger.success(f"Quantized model successfully exported to {onnx_weights}")

        except Exception as e:
            logger.error(f"Failed to process and export model artifacts: {e}")
            raise

    def generate_performance_report(
        self, precision: float, recall: float, latency_ms: float
    ) -> None:
        """
        Calculates required metrics (F1, FPS), validates them against thresholds, 
        and outputs the strict JSON file.

        Args:
            precision (float): The detection precision score (0.0 to 1.0).
            recall (float): The detection recall score (0.0 to 1.0).
            latency_ms (float): The average inference latency per frame in milliseconds.

        Returns:
            None
        """
        try:
            # Calculate derived metrics
            f1_score: float = 0.0
            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
                
            fps: float = 0.0
            if latency_ms > 0:
                fps = 1000.0 / latency_ms

            # Build the exact JSON structure required by the audit
            metrics: Dict[str, float] = {
                "detection_precision": round(precision, 3),
                "detection_recall": round(recall, 3),
                "f1_score": round(f1_score, 3),
                "average_fps_per_stream": round(fps, 1),
                "average_latency_ms": round(latency_ms, 1)
            }

            # Validation against strict audit thresholds
            passed: bool = True
            if metrics["detection_precision"] < 0.85:
                logger.warning(f"Precision {metrics['detection_precision']} is below 0.85 threshold.")
                passed = False
            if metrics["detection_recall"] < 0.80:
                logger.warning(f"Recall {metrics['detection_recall']} is below 0.80 threshold.")
                passed = False
            if metrics["f1_score"] < 0.85:
                logger.warning(f"F1-Score {metrics['f1_score']} is below 0.85 threshold.")
                passed = False
            if metrics["average_fps_per_stream"] < 15.0:
                logger.warning(f"FPS {metrics['average_fps_per_stream']} is below 15 threshold.")
                passed = False

            if passed:
                logger.success("All evaluation metrics passed the minimum audit thresholds!")

            # Write to JSON
            report_path: Path = self.reports_dir / "performance_metrics.json"
            with open(report_path, "w") as f:
                json.dump(metrics, f, indent=4)
                
            logger.info(f"Performance metrics successfully written to {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            raise