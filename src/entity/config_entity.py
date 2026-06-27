from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    validation_report_path: Path
    expected_columns: list[str]


@dataclass(frozen=True)
class ModelTrainingConfig:
    model_path: Path
    preprocessor_path: Path
    metrics_path: Path
    random_state: int
