from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.configuration import ConfigurationManager
from src.utils.common import load_object
from src.utils.exception import NetworkSecurityException
from src.utils.logger import logger


class PredictionPipeline:
    """Loads saved artifacts and generates intrusion predictions."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
        model_path: str | Path | None = None,
        required_feature_columns: list[str] | None = None,
    ) -> None:
        self.config_manager = (
            ConfigurationManager(config_path)
            if config_path is not None
            else ConfigurationManager()
        )
        self.preprocessor_path = Path(preprocessor_path or self.config_manager.preprocessor_path)
        self.model_path = Path(model_path or self.config_manager.model_path)
        self.required_feature_columns = required_feature_columns or self._get_required_feature_columns()

    def predict(self, input_data: pd.DataFrame | str | Path) -> pd.DataFrame:
        try:
            dataframe = self._load_input(input_data)
            self._validate_columns(dataframe)

            logger.info("Loading preprocessor from %s", self.preprocessor_path)
            preprocessor = load_object(self.preprocessor_path)

            logger.info("Loading model from %s", self.model_path)
            model_bundle = load_object(self.model_path)
            model, label_encoder = self._unpack_model_bundle(model_bundle)

            features = dataframe[self.required_feature_columns]
            transformed_features = preprocessor.transform(features)

            predictions = model.predict(transformed_features)
            prediction_labels = self._decode_predictions(predictions, label_encoder)

            result = dataframe.copy()
            result["prediction"] = prediction_labels

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_features)
                confidence_scores = probabilities.max(axis=1)
                result["confidence_score"] = confidence_scores
                result["attack_probability"] = self._attack_probability(
                    probabilities,
                    model,
                    label_encoder,
                )
                result["risk_level"] = [
                    self._risk_level(score) for score in result["attack_probability"]
                ]
            else:
                result["risk_level"] = "Medium"

            logger.info("Generated %s predictions", len(result))
            return result
        except Exception as error:
            raise NetworkSecurityException("Prediction pipeline failed") from error

    def predict_from_csv(self, csv_path: str | Path) -> pd.DataFrame:
        return self.predict(csv_path)

    def _get_required_feature_columns(self) -> list[str]:
        data_config = self.config_manager.data
        target_column = str(data_config.get("target_column", "label"))
        required_columns = list(data_config.get("required_columns", []))
        return [column for column in required_columns if column != target_column]

    @staticmethod
    def _load_input(input_data: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()

        csv_path = Path(input_data)
        if not csv_path.exists():
            raise FileNotFoundError(f"Prediction input CSV not found at {csv_path}")

        return pd.read_csv(csv_path)

    def _validate_columns(self, dataframe: pd.DataFrame) -> None:
        missing_columns = [
            column for column in self.required_feature_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")

    @staticmethod
    def _unpack_model_bundle(model_bundle: object) -> tuple[object, object | None]:
        if isinstance(model_bundle, dict) and "model" in model_bundle:
            return model_bundle["model"], model_bundle.get("label_encoder")
        return model_bundle, None

    @staticmethod
    def _decode_predictions(predictions: object, label_encoder: object | None) -> object:
        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            return label_encoder.inverse_transform(predictions)
        return predictions

    @staticmethod
    def _risk_level(confidence_score: float) -> str:
        if confidence_score < 0.60:
            return "Low"
        if confidence_score < 0.85:
            return "Medium"
        return "High"

    @staticmethod
    def _attack_probability(
        probabilities: object,
        model: object,
        label_encoder: object | None,
    ) -> object:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return probabilities.max(axis=1)

        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            class_labels = label_encoder.inverse_transform(classes)
        else:
            class_labels = classes

        for index, label in enumerate(class_labels):
            if str(label).lower() == "attack":
                return probabilities[:, index]

        return probabilities.max(axis=1)


def predict_from_csv(
    csv_path: str | Path,
    config_path: str | Path | None = None,
    preprocessor_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    pipeline = PredictionPipeline(
        config_path=config_path,
        preprocessor_path=preprocessor_path,
        model_path=model_path,
    )
    return pipeline.predict_from_csv(csv_path)
