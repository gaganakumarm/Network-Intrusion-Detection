from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.common import save_object
from src.utils.exception import NetworkSecurityException
from src.utils.logger import logger


class DataTransformation:
    """Builds and applies the preprocessing pipeline for model-ready arrays."""

    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    def run(self) -> DataTransformationArtifact:
        try:
            dataset_path = Path(self.config.processed_dataset_path)
            logger.info("Starting data transformation from %s", dataset_path)

            if not dataset_path.exists():
                raise FileNotFoundError(f"Processed dataset not found at {dataset_path}")

            dataframe = pd.read_csv(dataset_path)
            if self.config.target_column not in dataframe.columns:
                raise ValueError(f"Target column '{self.config.target_column}' not found")

            features = dataframe.drop(columns=[self.config.target_column])
            target = dataframe[self.config.target_column]

            numerical_columns = features.select_dtypes(include="number").columns.tolist()
            categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
            logger.info("Numerical columns: %s", numerical_columns)
            logger.info("Categorical columns: %s", categorical_columns)

            x_train, x_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            preprocessor = self._build_preprocessor(numerical_columns, categorical_columns)
            transformed_train_features = preprocessor.fit_transform(x_train)
            transformed_test_features = preprocessor.transform(x_test)

            train_array = self._combine_features_and_target(transformed_train_features, y_train)
            test_array = self._combine_features_and_target(transformed_test_features, y_test)

            preprocessor_path = Path(self.config.preprocessor_path)
            train_array_path = Path(self.config.train_array_path)
            test_array_path = Path(self.config.test_array_path)

            preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
            train_array_path.parent.mkdir(parents=True, exist_ok=True)
            test_array_path.parent.mkdir(parents=True, exist_ok=True)

            save_object(preprocessor_path, preprocessor)
            np.save(train_array_path, train_array)
            np.save(test_array_path, test_array)

            logger.info("Saved preprocessor to %s", preprocessor_path)
            logger.info("Saved train array to %s", train_array_path)
            logger.info("Saved test array to %s", test_array_path)

            return DataTransformationArtifact(
                preprocessor_path=preprocessor_path,
                train_array_path=train_array_path,
                test_array_path=test_array_path,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                train_rows=train_array.shape[0],
                test_rows=test_array.shape[0],
            )
        except Exception as error:
            raise NetworkSecurityException("Data transformation failed") from error

    @staticmethod
    def _build_preprocessor(
        numerical_columns: list[str],
        categorical_columns: list[str],
    ) -> ColumnTransformer:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ]
        )

    @staticmethod
    def _combine_features_and_target(features: np.ndarray, target: pd.Series) -> np.ndarray:
        dense_features = features.toarray() if hasattr(features, "toarray") else features
        return np.column_stack((dense_features, target.to_numpy()))
