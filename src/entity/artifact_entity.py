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
class ModelTrainerArtifact:
    trained_model_path: Path
    preprocessor_path: Path
    metrics_path: Path
    test_score: float
