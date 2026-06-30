from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    dataset_path: Path
    processed_dataset_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    dataset_path: Path
    validation_report_path: Path
    required_columns: list[str]
    target_column: str


@dataclass(frozen=True)
class ModelTrainingConfig:
    model_path: Path
    preprocessor_path: Path
    metrics_path: Path
    random_state: int
