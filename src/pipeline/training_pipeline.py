from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_training import ModelTrainer
from src.config.configuration import ConfigurationManager
from src.entity.artifact_entity import ModelTrainerArtifact
from src.utils.exception import NetworkSecurityException
from src.utils.logger import logger


class TrainingPipeline:
    """Runs the complete training workflow from raw CSV to trained model."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_manager = (
            ConfigurationManager(config_path)
            if config_path is not None
            else ConfigurationManager()
        )

    def run(self) -> ModelTrainerArtifact:
        try:
            logger.info("========== Training pipeline started ==========")

            logger.info("Stage 1: Data ingestion started")
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            ingestion_artifact = DataIngestion(data_ingestion_config).run()
            logger.info("Stage 1: Data ingestion completed: %s", ingestion_artifact)

            logger.info("Stage 2: Data validation started")
            data_validation_config = self.config_manager.get_data_validation_config()
            validation_artifact = DataValidation(data_validation_config).run()
            logger.info("Stage 2: Data validation completed: %s", validation_artifact)

            if not validation_artifact.validation_status:
                raise ValueError(
                    f"Data validation failed. See report: {validation_artifact.validation_report_path}"
                )

            logger.info("Stage 3: Data transformation started")
            data_transformation_config = self.config_manager.get_data_transformation_config()
            transformation_artifact = DataTransformation(data_transformation_config).run()
            logger.info("Stage 3: Data transformation completed: %s", transformation_artifact)

            logger.info("Stage 4: Model training started")
            model_training_config = self.config_manager.get_model_training_config()
            model_artifact = ModelTrainer(model_training_config).run()
            logger.info("Stage 4: Model training completed: %s", model_artifact)

            logger.info("========== Training pipeline completed ==========")
            print(f"Final model path: {model_artifact.trained_model_path}")
            print(f"Metrics path: {model_artifact.metrics_path}")

            return model_artifact
        except Exception as error:
            logger.exception("Training pipeline failed")
            raise NetworkSecurityException("Training pipeline failed") from error


if __name__ == "__main__":
    TrainingPipeline().run()
