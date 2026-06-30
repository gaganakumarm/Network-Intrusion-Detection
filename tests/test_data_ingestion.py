from pathlib import Path

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig


def test_data_ingestion_loads_dataset_and_creates_processed_copy(tmp_path: Path) -> None:
    dataset_path = tmp_path / "raw" / "sample.csv"
    processed_dataset_path = tmp_path / "processed" / "sample.csv"
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

    config = DataIngestionConfig(
        dataset_path=dataset_path,
        processed_dataset_path=processed_dataset_path,
    )

    artifact = DataIngestion(config).run()

    assert isinstance(artifact, DataIngestionArtifact)
    assert artifact.processed_dataset_path.exists()
    assert artifact.rows == 2
    assert artifact.columns == 6

    processed_data = pd.read_csv(artifact.processed_dataset_path)
    assert processed_data.equals(sample_data)
