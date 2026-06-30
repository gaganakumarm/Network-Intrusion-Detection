from pathlib import Path

import pandas as pd

from src.components.data_validation import DataValidation
from src.entity.artifact_entity import DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.utils.common import load_json


REQUIRED_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "src_bytes",
    "dst_bytes",
    "label",
]


def test_data_validation_creates_report_and_passes_valid_data(tmp_path: Path) -> None:
    dataset_path = tmp_path / "processed" / "sample.csv"
    validation_report_path = tmp_path / "reports" / "validation_report.json"
    dataset_path.parent.mkdir(parents=True)

    sample_data = pd.DataFrame(
        {
            "duration": [0, 2],
            "protocol_type": ["tcp", "udp"],
            "service": ["http", "dns"],
            "src_bytes": [181, 239],
            "dst_bytes": [5450, 486],
            "label": ["normal", "attack"],
        }
    )
    sample_data.to_csv(dataset_path, index=False)

    config = DataValidationConfig(
        dataset_path=dataset_path,
        validation_report_path=validation_report_path,
        required_columns=REQUIRED_COLUMNS,
        target_column="label",
    )

    artifact = DataValidation(config).run()
    report = load_json(validation_report_path)

    assert isinstance(artifact, DataValidationArtifact)
    assert artifact.validation_status is True
    assert artifact.validation_report_path.exists()
    assert report["validation_status"] is True
    assert report["file_exists"] is True
    assert report["csv_readable"] is True
    assert report["required_columns_exist"] is True
    assert report["target_column_exists"] is True
    assert report["duplicate_rows"] == 0
    assert all(count == 0 for count in report["missing_values"].values())
