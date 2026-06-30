from pathlib import Path

import numpy as np

from src.components.model_training import ModelTraining
from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import ModelTrainingConfig
from src.utils.common import load_json, load_object


def test_model_training_saves_best_model_and_metrics(tmp_path: Path) -> None:
    train_array_path = tmp_path / "processed" / "train.npy"
    test_array_path = tmp_path / "processed" / "test.npy"
    model_path = tmp_path / "artifacts" / "model.pkl"
    metrics_path = tmp_path / "reports" / "metrics.json"
    train_array_path.parent.mkdir(parents=True)

    train_array = np.array(
        [
            [0.1, 1.0, "normal"],
            [0.2, 1.1, "normal"],
            [0.3, 1.2, "normal"],
            [3.0, 4.0, "attack"],
            [3.1, 4.1, "attack"],
            [3.2, 4.2, "attack"],
            [0.4, 1.3, "normal"],
            [3.3, 4.3, "attack"],
        ],
        dtype=object,
    )
    test_array = np.array(
        [
            [0.15, 1.05, "normal"],
            [3.15, 4.15, "attack"],
            [0.35, 1.25, "normal"],
            [3.35, 4.35, "attack"],
        ],
        dtype=object,
    )

    np.save(train_array_path, train_array)
    np.save(test_array_path, test_array)

    config = ModelTrainingConfig(
        train_array_path=train_array_path,
        test_array_path=test_array_path,
        model_path=model_path,
        metrics_path=metrics_path,
        random_state=42,
    )

    artifact = ModelTraining(config).run()
    model_bundle = load_object(model_path)
    metrics = load_json(metrics_path)

    assert isinstance(artifact, ModelTrainerArtifact)
    assert artifact.trained_model_path.exists()
    assert artifact.metrics_path.exists()
    assert artifact.best_model_name in metrics
    assert artifact.test_score == metrics[artifact.best_model_name]["f1_score"]

    assert "model" in model_bundle
    assert "label_encoder" in model_bundle
    assert set(metrics.keys()) == {
        "LogisticRegression",
        "RandomForestClassifier",
        "XGBClassifier",
    }
    for model_metrics in metrics.values():
        assert set(model_metrics.keys()) == {"accuracy", "precision", "recall", "f1_score"}
        assert all(0.0 <= score <= 1.0 for score in model_metrics.values())
