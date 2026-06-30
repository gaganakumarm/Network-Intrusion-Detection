from pathlib import Path

import numpy as np
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.common import load_object


def test_data_transformation_creates_preprocessor_and_arrays(tmp_path: Path) -> None:
    dataset_path = tmp_path / "processed" / "sample.csv"
    preprocessor_path = tmp_path / "artifacts" / "preprocessor.pkl"
    train_array_path = tmp_path / "processed" / "train.npy"
    test_array_path = tmp_path / "processed" / "test.npy"
    dataset_path.parent.mkdir(parents=True)

    sample_data = pd.DataFrame(
        {
            "duration": [0, 2, 5, 1, 3],
            "protocol_type": ["tcp", "udp", "tcp", "icmp", "udp"],
            "service": ["http", "dns", "ftp", "http", "smtp"],
            "src_bytes": [181, 239, 300, 120, 500],
            "dst_bytes": [5450, 486, 1000, 230, 900],
            "label": ["normal", "attack", "normal", "attack", "normal"],
        }
    )
    sample_data.to_csv(dataset_path, index=False)

    config = DataTransformationConfig(
        processed_dataset_path=dataset_path,
        preprocessor_path=preprocessor_path,
        train_array_path=train_array_path,
        test_array_path=test_array_path,
        target_column="label",
        test_size=0.4,
        random_state=42,
    )

    artifact = DataTransformation(config).run()

    assert isinstance(artifact, DataTransformationArtifact)
    assert artifact.preprocessor_path.exists()
    assert artifact.train_array_path.exists()
    assert artifact.test_array_path.exists()
    assert artifact.numerical_columns == ["duration", "src_bytes", "dst_bytes"]
    assert artifact.categorical_columns == ["protocol_type", "service"]
    assert artifact.train_rows == 3
    assert artifact.test_rows == 2

    preprocessor = load_object(preprocessor_path)
    train_array = np.load(train_array_path, allow_pickle=True)
    test_array = np.load(test_array_path, allow_pickle=True)

    assert preprocessor is not None
    assert train_array.shape[0] == 3
    assert test_array.shape[0] == 2
    assert train_array.shape[1] == test_array.shape[1]
