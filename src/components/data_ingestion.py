from pathlib import Path

import pandas as pd

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.utils.exception import NetworkSecurityException
from src.utils.logger import logger


class DataIngestion:
    """Loads the raw dataset and stores a processed copy."""

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def run(self) -> DataIngestionArtifact:
        try:
            dataset_path = Path(self.config.dataset_path)
            processed_dataset_path = Path(self.config.processed_dataset_path)

            logger.info("Starting data ingestion from %s", dataset_path)

            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            dataframe = pd.read_csv(dataset_path)
            logger.info("Loaded dataset with %s rows and %s columns", dataframe.shape[0], dataframe.shape[1])

            processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_csv(processed_dataset_path, index=False)
            logger.info("Saved processed dataset to %s", processed_dataset_path)

            return DataIngestionArtifact(
                processed_dataset_path=processed_dataset_path,
                rows=dataframe.shape[0],
                columns=dataframe.shape[1],
            )
        except Exception as error:
            raise NetworkSecurityException("Data ingestion failed") from error
