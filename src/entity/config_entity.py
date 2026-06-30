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
class DataTransformationConfig:
    processed_dataset_path: Path
    preprocessor_path: Path
    train_array_path: Path
    test_array_path: Path
    target_column: str
    test_size: float
    random_state: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    train_array_path: Path
    test_array_path: Path
    model_path: Path
    metrics_path: Path
    random_state: int
