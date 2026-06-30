from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.pipeline.prediction_pipeline import PredictionPipeline, predict_from_csv
from src.utils.common import save_object


FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "src_bytes",
    "dst_bytes",
]


def _create_prediction_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
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
    x_train = training_data[FEATURE_COLUMNS]
    y_train = label_encoder.fit_transform(training_data["label"])
    transformed_features = preprocessor.fit_transform(x_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(transformed_features, y_train)

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

    return config_path, preprocessor_path, model_path


def test_prediction_pipeline_predicts_from_dataframe(tmp_path: Path) -> None:
    config_path, preprocessor_path, model_path = _create_prediction_artifacts(tmp_path)
    input_data = pd.DataFrame(
        {
            "duration": [0, 9],
            "protocol_type": ["tcp", "icmp"],
            "service": ["http", "smtp"],
            "src_bytes": [110, 930],
            "dst_bytes": [210, 830],
        }
    )

    pipeline = PredictionPipeline(
        config_path=config_path,
        preprocessor_path=preprocessor_path,
        model_path=model_path,
    )
    predictions = pipeline.predict(input_data)

    assert list(predictions["prediction"]) == ["normal", "attack"]
    assert "confidence_score" in predictions.columns
    assert "attack_probability" in predictions.columns
    assert "risk_level" in predictions.columns
    assert predictions["confidence_score"].between(0, 1).all()
    assert predictions["attack_probability"].between(0, 1).all()
    assert set(predictions["risk_level"]).issubset({"Low", "Medium", "High"})


def test_predict_from_csv_helper(tmp_path: Path) -> None:
    config_path, preprocessor_path, model_path = _create_prediction_artifacts(tmp_path)
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "duration": [1],
            "protocol_type": ["udp"],
            "service": ["dns"],
            "src_bytes": [125],
            "dst_bytes": [225],
        }
    ).to_csv(csv_path, index=False)

    predictions = predict_from_csv(
        csv_path,
        config_path=config_path,
        preprocessor_path=preprocessor_path,
        model_path=model_path,
    )

    assert len(predictions) == 1
    assert predictions.loc[0, "prediction"] == "normal"
    assert "confidence_score" in predictions.columns
    assert "attack_probability" in predictions.columns
