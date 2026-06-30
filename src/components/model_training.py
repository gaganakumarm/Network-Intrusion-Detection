from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import ModelTrainingConfig
from src.utils.common import save_json, save_object
from src.utils.exception import NetworkSecurityException
from src.utils.logger import logger


class ModelTrainer:
    """Trains candidate classifiers and saves the best model by F1-score."""

    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config = config

    def run(self) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training")

            train_array = self._load_array(self.config.train_array_path)
            test_array = self._load_array(self.config.test_array_path)

            x_train, y_train = self._split_features_and_target(train_array)
            x_test, y_test = self._split_features_and_target(test_array)

            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            models = self._get_models()
            metrics: dict[str, dict[str, float]] = {}
            best_model_name = ""
            best_model = None
            best_f1_score = -1.0

            for model_name, model in models.items():
                logger.info("Training %s", model_name)
                model.fit(x_train, y_train_encoded)
                predictions = model.predict(x_test)

                model_metrics = self._evaluate(y_test_encoded, predictions)
                metrics[model_name] = model_metrics
                logger.info("%s metrics: %s", model_name, model_metrics)

                if model_metrics["f1_score"] > best_f1_score:
                    best_f1_score = model_metrics["f1_score"]
                    best_model_name = model_name
                    best_model = model

            if best_model is None:
                raise ValueError("No model was trained successfully")

            model_path = Path(self.config.model_path)
            metrics_path = Path(self.config.metrics_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            model_bundle = {
                "model": best_model,
                "model_name": best_model_name,
                "label_encoder": label_encoder,
            }

            save_object(model_path, model_bundle)
            save_json(metrics_path, metrics)

            logger.info("Best model: %s with F1-score %.4f", best_model_name, best_f1_score)
            logger.info("Saved best model to %s", model_path)
            logger.info("Saved metrics to %s", metrics_path)

            return ModelTrainerArtifact(
                trained_model_path=model_path,
                metrics_path=metrics_path,
                best_model_name=best_model_name,
                test_score=best_f1_score,
                metrics=metrics,
            )
        except Exception as error:
            raise NetworkSecurityException("Model training failed") from error

    @staticmethod
    def _load_array(path: str | Path) -> np.ndarray:
        array_path = Path(path)
        if not array_path.exists():
            raise FileNotFoundError(f"Array file not found at {array_path}")
        return np.load(array_path, allow_pickle=True)

    @staticmethod
    def _split_features_and_target(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("Expected a 2D array with features and target column")

        features = array[:, :-1].astype(float)
        target = array[:, -1]
        return features, target

    def _get_models(self) -> dict[str, object]:
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.config.random_state),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=50,
                random_state=self.config.random_state,
                n_jobs=1,
            ),
            "XGBClassifier": XGBClassifier(
                n_estimators=25,
                max_depth=3,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=self.config.random_state,
                n_jobs=1,
            ),
        }

    @staticmethod
    def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }


ModelTraining = ModelTrainer
