import logging
from pathlib import Path

from src.config.configuration import ConfigurationManager
from src.constants import CONFIG_FILE_PATH
from src.utils.common import (
    create_directories,
    load_json,
    load_object,
    read_yaml,
    save_json,
    save_object,
)
from src.utils.logger import LOG_FILE, logger


def test_logger_creates_log_file() -> None:
    logger.info("Testing logger setup")

    assert LOG_FILE.parent == Path("logs")
    assert LOG_FILE.exists()
    assert logger.level in (logging.NOTSET, logging.INFO)


def test_read_yaml_loads_config() -> None:
    config = read_yaml(CONFIG_FILE_PATH)

    assert config["random_state"] == 42
    assert config["data"]["raw_dir"] == "data/raw"
    assert config["model"]["path"] == "artifacts/model_trainer/model.pkl"


def test_configuration_manager_exposes_values() -> None:
    config = ConfigurationManager()

    assert config.random_state == 42
    assert config.model_path == Path("artifacts/model_trainer/model.pkl")
    assert config.preprocessor_path == Path("artifacts/model_trainer/preprocessor.pkl")
    assert config.metrics_path == Path("reports/metrics.json")


def test_create_directories(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "nested" / "second"

    create_directories([first_dir, second_dir])

    assert first_dir.exists()
    assert second_dir.exists()


def test_json_save_and_load(tmp_path: Path) -> None:
    json_path = tmp_path / "sample.json"
    data = {"accuracy": 0.95, "model": "baseline"}

    save_json(json_path, data)
    loaded_data = load_json(json_path)

    assert loaded_data == data


def test_object_save_and_load(tmp_path: Path) -> None:
    object_path = tmp_path / "sample.pkl"
    data = {"features": ["duration", "protocol_type"], "count": 2}

    save_object(object_path, data)
    loaded_object = load_object(object_path)

    assert loaded_object == data
