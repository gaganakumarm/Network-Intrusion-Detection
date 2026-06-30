from pathlib import Path

import pandas as pd

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import load_json, load_object


def test_training_pipeline_runs_end_to_end_with_sample_data(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "raw" / "network_intrusion.csv"
    processed_dataset_path = tmp_path / "data" / "processed" / "network_intrusion.csv"
    train_array_path = tmp_path / "data" / "processed" / "train.npy"
    test_array_path = tmp_path / "data" / "processed" / "test.npy"
    validation_report_path = tmp_path / "reports" / "validation_report.json"
    metrics_path = tmp_path / "reports" / "metrics.json"
    preprocessor_path = tmp_path / "artifacts" / "preprocessor.pkl"
    model_path = tmp_path / "artifacts" / "model.pkl"
    config_path = tmp_path / "config.yaml"

    dataset_path.parent.mkdir(parents=True)
    sample_data = pd.DataFrame(
        {
            "duration": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "protocol_type": ["tcp", "udp", "tcp", "icmp", "udp", "tcp", "icmp", "udp", "tcp", "icmp"],
            "service": ["http", "dns", "ftp", "smtp", "ssh", "http", "dns", "ftp", "smtp", "ssh"],
            "src_bytes": [100, 120, 150, 800, 850, 900, 160, 180, 920, 940],
            "dst_bytes": [200, 220, 250, 700, 750, 760, 260, 280, 780, 790],
            "label": ["normal", "normal", "normal", "attack", "attack", "attack", "normal", "normal", "attack", "attack"],
        }
    )
    sample_data.to_csv(dataset_path, index=False)

    config_path.write_text(
        f"""
data:
  root_dir: {tmp_path / "data"}
  raw_dir: {tmp_path / "data" / "raw"}
  dataset_path: {dataset_path}
  processed_dir: {tmp_path / "data" / "processed"}
  processed_dataset_path: {processed_dataset_path}
  train_file: {tmp_path / "data" / "processed" / "train.csv"}
  test_file: {tmp_path / "data" / "processed" / "test.csv"}
  train_array_path: {train_array_path}
  test_array_path: {test_array_path}
  target_column: label
  required_columns:
    - duration
    - protocol_type
    - service
    - src_bytes
    - dst_bytes
    - label

artifacts:
  root_dir: {tmp_path / "artifacts"}
  data_ingestion_dir: {tmp_path / "artifacts" / "data_ingestion"}
  data_validation_dir: {tmp_path / "artifacts" / "data_validation"}
  model_trainer_dir: {tmp_path / "artifacts" / "model_trainer"}

reports:
  root_dir: {tmp_path / "reports"}
  validation_report: {validation_report_path}
  validation_report_path: {validation_report_path}
  metrics_file: {metrics_path}

model:
  path: {model_path}

preprocessor:
  path: {preprocessor_path}

metrics:
  path: {metrics_path}

random_state: 42
test_size: 0.3
""",
        encoding="utf-8",
    )

    artifact = TrainingPipeline(config_path=config_path).run()

    assert processed_dataset_path.exists()
    assert validation_report_path.exists()
    assert preprocessor_path.exists()
    assert train_array_path.exists()
    assert test_array_path.exists()
    assert artifact.trained_model_path.exists()
    assert artifact.metrics_path.exists()

    validation_report = load_json(validation_report_path)
    metrics = load_json(metrics_path)
    model_bundle = load_object(model_path)

    assert validation_report["validation_status"] is True
    assert artifact.best_model_name in metrics
    assert "model" in model_bundle
    assert "label_encoder" in model_bundle
