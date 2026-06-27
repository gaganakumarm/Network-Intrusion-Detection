import json
import pickle
from pathlib import Path
from typing import Any

import yaml

from src.utils.exception import NetworkSecurityException


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary."""
    try:
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
        return content or {}
    except Exception as error:
        raise NetworkSecurityException(f"Failed to read YAML file: {path}") from error


def create_directories(paths: list[str | Path]) -> None:
    """Create each directory in the provided list if it does not already exist."""
    try:
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as error:
        raise NetworkSecurityException("Failed to create directories") from error


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """Save dictionary data to a JSON file."""
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as error:
        raise NetworkSecurityException(f"Failed to save JSON file: {path}") from error


def load_json(path: str | Path) -> dict[str, Any]:
    """Load dictionary data from a JSON file."""
    try:
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except Exception as error:
        raise NetworkSecurityException(f"Failed to load JSON file: {path}") from error


def save_object(path: str | Path, obj: Any) -> None:
    """Save a Python object with pickle."""
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as object_file:
            pickle.dump(obj, object_file)
    except Exception as error:
        raise NetworkSecurityException(f"Failed to save object: {path}") from error


def load_object(path: str | Path) -> Any:
    """Load a Python object saved with pickle."""
    try:
        file_path = Path(path)
        with file_path.open("rb") as object_file:
            return pickle.load(object_file)
    except Exception as error:
        raise NetworkSecurityException(f"Failed to load object: {path}") from error
