from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path


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
