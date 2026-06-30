from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    processed_dataset_path: Path
    rows: int
    columns: int


@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: Path
    message: str


@dataclass(frozen=True)
class DataTransformationArtifact:
    preprocessor_path: Path
    train_array_path: Path
    test_array_path: Path
    numerical_columns: list[str]
    categorical_columns: list[str]
    train_rows: int
    test_rows: int


@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_path: Path
    metrics_path: Path
    best_model_name: str
    test_score: float
    metrics: dict[str, dict[str, float]]
