from pathlib import Path
from typing import Any

from src.constants import CONFIG_FILE_PATH
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
