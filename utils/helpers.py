import os
import json
import joblib
import yaml
from pathlib import Path
from typing import Any, Dict
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Ensure that the directory exists; create if it does not."""
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise CustomException(f"Failed to create directory {path}", e)


def read_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Read a YAML file and return the configuration dictionary."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML configuration file not found at: {file_path}")
        
        with open(path, "r", encoding="utf-8") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"Successfully loaded YAML configuration from: {file_path}")
            return config if config is not None else {}
    except Exception as e:
        logger.error(f"Error while reading YAML file: {file_path}")
        raise CustomException(f"Failed to read YAML file {file_path}", e)


def save_object(file_path: str | Path, obj: Any) -> None:
    """Save a Python object using joblib."""
    try:
        path = Path(file_path)
        ensure_dir(path.parent)
        joblib.dump(obj, path)
        logger.info(f"Successfully saved object to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save object to {file_path}: {e}")
        raise CustomException(f"Failed to save object to {file_path}", e)


def load_object(file_path: str | Path) -> Any:
    """Load a Python object using joblib."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {file_path}")
        obj = joblib.load(path)
        logger.info(f"Successfully loaded object from: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load object from {file_path}: {e}")
        raise CustomException(f"Failed to load object from {file_path}", e)


def save_json(file_path: str | Path, data: Dict[str, Any]) -> None:
    """Save dictionary data to a JSON file."""
    try:
        path = Path(file_path)
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved JSON to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise CustomException(f"Failed to save JSON to {file_path}", e)


def load_json(file_path: str | Path) -> Dict[str, Any]:
    """Load JSON data from a file."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found at: {file_path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        raise CustomException(f"Failed to load JSON from {file_path}", e)
