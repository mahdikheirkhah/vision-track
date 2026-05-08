import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.analysis import MetricsEvaluator

# Assuming the class above is saved in utils/analysis.py
# from utils.analysis import MetricsEvaluator

def test_metrics_evaluator_init(tmp_path: Path) -> None:
    """Tests proper directory initialization."""
    run_dir = tmp_path / "runs"
    chk_dir = tmp_path / "models/checkpoints"
    rep_dir = tmp_path / "reports"
    
    evaluator = MetricsEvaluator(str(run_dir), str(chk_dir), str(rep_dir))
    
    assert evaluator.checkpoints_dir.exists()
    assert evaluator.reports_dir.exists()


def test_generate_performance_report_success(tmp_path: Path) -> None:
    """Tests that math calculations (F1, FPS) and JSON formatting are correct."""
    rep_dir = tmp_path / "reports"
    evaluator = MetricsEvaluator(reports_dir=str(rep_dir))
    
    # Inputs: Precision 0.90, Recall 0.85, Latency 50ms (Expected FPS: 20)
    evaluator.generate_performance_report(precision=0.90, recall=0.85, latency_ms=50.0)
    
    json_file = rep_dir / "performance_metrics.json"
    assert json_file.exists()
    
    with open(json_file, "r") as f:
        data = json.load(f)
        
    assert data["detection_precision"] == 0.9
    assert data["detection_recall"] == 0.85
    # F1 = 2 * (0.9 * 0.85) / (1.75) ≈ 0.874
    assert data["f1_score"] == 0.874
    assert data["average_fps_per_stream"] == 20.0
    assert data["average_latency_ms"] == 50.0


def test_process_and_export_artifacts_failure(tmp_path: Path) -> None:
    """Tests exception handling if the trained weights are missing."""
    evaluator = MetricsEvaluator(run_dir=str(tmp_path))
    
    with pytest.raises(FileNotFoundError, match="Trained weights not found"):
        evaluator.process_and_export_artifacts()