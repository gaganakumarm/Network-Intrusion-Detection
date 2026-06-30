from pathlib import Path
from typing import Any

import pandas as pd

from src.entity.artifact_entity import DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.utils.common import save_json
from src.utils.exception import NetworkSecurityException
from src.utils.logger import logger


class DataValidation:
    """Validates the processed dataset and writes a JSON report."""

    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    def run(self) -> DataValidationArtifact:
        try:
            dataset_path = Path(self.config.dataset_path)
            report_path = Path(self.config.validation_report_path)

            logger.info("Starting data validation for %s", dataset_path)

            report = self._build_report(dataset_path)
            save_json(report_path, report)

            validation_status = bool(report["validation_status"])
            message = "Validation passed" if validation_status else "Validation failed"

            logger.info("%s. Report saved to %s", message, report_path)

            return DataValidationArtifact(
                validation_status=validation_status,
                validation_report_path=report_path,
                message=message,
            )
        except Exception as error:
            raise NetworkSecurityException("Data validation failed") from error

    def _build_report(self, dataset_path: Path) -> dict[str, Any]:
        report: dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "file_exists": dataset_path.exists(),
            "csv_readable": False,
            "required_columns_exist": False,
            "target_column_exists": False,
            "duplicate_rows": 0,
            "missing_values": {},
            "data_types": {},
            "validation_status": False,
            "errors": [],
        }

        if not dataset_path.exists():
            report["errors"].append(f"File does not exist: {dataset_path}")
            return report

        try:
            dataframe = pd.read_csv(dataset_path)
            report["csv_readable"] = True
        except Exception as error:
            report["errors"].append(f"CSV is not readable: {error}")
            return report

        missing_columns = [
            column for column in self.config.required_columns if column not in dataframe.columns
        ]
        report["required_columns_exist"] = len(missing_columns) == 0
        if missing_columns:
            report["errors"].append(f"Missing required columns: {missing_columns}")

        report["target_column_exists"] = self.config.target_column in dataframe.columns
        if not report["target_column_exists"]:
            report["errors"].append(f"Missing target column: {self.config.target_column}")

        duplicate_rows = int(dataframe.duplicated().sum())
        missing_values = {
            column: int(count) for column, count in dataframe.isna().sum().to_dict().items()
        }

        report["duplicate_rows"] = duplicate_rows
        report["missing_values"] = missing_values
        report["data_types"] = {column: str(dtype) for column, dtype in dataframe.dtypes.items()}

        if duplicate_rows > 0:
            report["errors"].append(f"Duplicate rows found: {duplicate_rows}")

        columns_with_missing_values = [
            column for column, count in missing_values.items() if count > 0
        ]
        if columns_with_missing_values:
            report["errors"].append(f"Missing values found in columns: {columns_with_missing_values}")

        report["validation_status"] = (
            report["file_exists"]
            and report["csv_readable"]
            and report["required_columns_exist"]
            and report["target_column_exists"]
            and duplicate_rows == 0
            and not columns_with_missing_values
        )
        return report
