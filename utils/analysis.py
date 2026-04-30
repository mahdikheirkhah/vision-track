import pandas as pd
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
from loguru import logger

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