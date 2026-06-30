from pathlib import Path
from typing import Any

from src.constants import CONFIG_FILE_PATH
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.utils.common import read_yaml


class ConfigurationManager:
    """Loads project configuration values from config.yaml."""

    def __init__(self, config_path: str | Path = CONFIG_FILE_PATH) -> None:
        self.config_path = Path(config_path)
        self.config = read_yaml(self.config_path)

    @property
    def data(self) -> dict[str, Any]:
        return self.config.get("data", {})

    @property
    def artifacts(self) -> dict[str, Any]:
        return self.config.get("artifacts", {})

    @property
    def reports(self) -> dict[str, Any]:
        return self.config.get("reports", {})

    @property
    def model_path(self) -> Path:
        return Path(self.config.get("model", {}).get("path", "artifacts/model.pkl"))

    @property
    def preprocessor_path(self) -> Path:
        return Path(self.config.get("preprocessor", {}).get("path", "artifacts/preprocessor.pkl"))

    @property
    def metrics_path(self) -> Path:
        return Path(self.config.get("metrics", {}).get("path", "reports/metrics.json"))

    @property
    def random_state(self) -> int:
        return int(self.config.get("random_state", 42))

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_config = self.data
        return DataIngestionConfig(
            dataset_path=Path(data_config["dataset_path"]),
            processed_dataset_path=Path(data_config["processed_dataset_path"]),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        data_config = self.data
        reports_config = self.reports
        return DataValidationConfig(
            dataset_path=Path(data_config["processed_dataset_path"]),
            validation_report_path=Path(reports_config["validation_report_path"]),
            required_columns=list(data_config["required_columns"]),
            target_column=str(data_config["target_column"]),
        )
