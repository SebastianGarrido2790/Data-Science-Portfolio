"""
Common utility functions for configuration management, directory orchestration,
and data serialization to support reproducible MLOps pipelines.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from src.constants import PROJECT_ROOT
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Returns:
        ConfigBox: A dictionary-like object that allows dot notation access
                   (e.g., config.key instead of config['key']).

    Raises:
        ValueError: If the YAML file is empty.
        e: If there is a parsing error.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(
                f"yaml file: {Path(path_to_yaml).resolve().relative_to(PROJECT_ROOT)} loaded successfully"
            )
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise CustomException(e, sys)


def create_directories(path_to_directories: list, verbose: bool = True) -> None:
    """
    Creates a list of directories.

    Args:
        path_to_directories (list): List of paths to create.
        verbose (bool, optional): If True, logs the creation. Defaults to True.
    """
    for path in path_to_directories:
        path = Path(path).resolve()
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path.relative_to(PROJECT_ROOT)}")


def save_json(path: Path, data: dict) -> None:
    """
    Saves a dictionary to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {Path(path).resolve().relative_to(PROJECT_ROOT)}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns a ConfigBox.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(
        f"json file loaded successfully from: {Path(path).resolve().relative_to(PROJECT_ROOT)}"
    )
    return ConfigBox(content)


def save_bin(data: Any, path: Path) -> None:
    """
    Saves data as a binary file (pickle) using joblib.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {Path(path).resolve().relative_to(PROJECT_ROOT)}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file.
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {Path(path).resolve().relative_to(PROJECT_ROOT)}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
