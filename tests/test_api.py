from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from api.dependencies import get_prediction_pipeline
from api.main import app
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import save_object


FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "src_bytes",
    "dst_bytes",
]


def _create_prediction_pipeline(tmp_path: Path) -> PredictionPipeline:
    model_path = tmp_path / "artifacts" / "model.pkl"
    preprocessor_path = tmp_path / "artifacts" / "preprocessor.pkl"
    config_path = tmp_path / "config.yaml"

    training_data = pd.DataFrame(
        {
            "duration": [0, 1, 2, 8, 9, 10],
            "protocol_type": ["tcp", "udp", "tcp", "icmp", "udp", "icmp"],
            "service": ["http", "dns", "ftp", "smtp", "ssh", "smtp"],
            "src_bytes": [100, 120, 140, 900, 920, 940],
            "dst_bytes": [200, 220, 240, 800, 820, 840],
            "label": ["normal", "normal", "normal", "attack", "attack", "attack"],
        }
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["duration", "src_bytes", "dst_bytes"],
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["protocol_type", "service"],
            ),
        ]
    )

    label_encoder = LabelEncoder()
    transformed_features = preprocessor.fit_transform(training_data[FEATURE_COLUMNS])
    encoded_target = label_encoder.fit_transform(training_data["label"])

    model = LogisticRegression(max_iter=1000)
    model.fit(transformed_features, encoded_target)

    save_object(preprocessor_path, preprocessor)
    save_object(
        model_path,
        {
            "model": model,
            "model_name": "LogisticRegression",
            "label_encoder": label_encoder,
        },
    )

    config_path.write_text(
        f"""
data:
  required_columns:
    - duration
    - protocol_type
    - service
    - src_bytes
    - dst_bytes
    - label
  target_column: label

model:
  path: {model_path}

preprocessor:
  path: {preprocessor_path}
""",
        encoding="utf-8",
    )

    return PredictionPipeline(config_path=config_path)


def _client_with_pipeline(tmp_path: Path) -> TestClient:
    pipeline = _create_prediction_pipeline(tmp_path)
    app.dependency_overrides[get_prediction_pipeline] = lambda: pipeline
    return TestClient(app)


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint(tmp_path: Path) -> None:
    client = _client_with_pipeline(tmp_path)

    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_exists"] is True
    assert body["preprocessor_exists"] is True
    assert body["model_version"] == "1.0.0"
    assert body["model_name"] == "LogisticRegression"
    assert body["required_feature_columns"] == FEATURE_COLUMNS


def test_predict_endpoint(tmp_path: Path) -> None:
    client = _client_with_pipeline(tmp_path)

    response = client.post(
        "/predict",
        json={
            "duration": 0,
            "protocol_type": "tcp",
            "service": "http",
            "src_bytes": 110,
            "dst_bytes": 210,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "normal"
    assert 0.0 <= body["attack_probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["model_version"] == "1.0.0"
    assert body["prediction_timestamp"].endswith("Z")


def test_predict_csv_endpoint(tmp_path: Path) -> None:
    client = _client_with_pipeline(tmp_path)
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "duration": [0, 9],
            "protocol_type": ["tcp", "icmp"],
            "service": ["http", "smtp"],
            "src_bytes": [110, 930],
            "dst_bytes": [210, 830],
        }
    ).to_csv(csv_path, index=False)

    with csv_path.open("rb") as csv_file:
        response = client.post(
            "/predict-csv",
            files={"file": ("input.csv", csv_file, "text/csv")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["total_records"] == 2
    assert body["attack_count"] == 1
    assert body["normal_count"] == 1
    assert len(body["prediction_table"]) == 2
    assert "attack_probability" in body["prediction_table"][0]


def teardown_function() -> None:
    app.dependency_overrides.clear()
