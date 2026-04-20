"""
Common utility functions for the NYC Taxi Tip Prediction system.

This module provides reusable helper functions for handling YAML configuration,
directory management, JSON serialization, and file metadata. These utilities
ensure consistency across different pipeline stages.
"""

import os
import sys
from pathlib import Path

import yaml

from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def read_yaml(path_to_yaml: Path) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.

    Args:
        path_to_yaml (Path): Direct path to the YAML file to be read.

    Raises:
        CustomExceptionError: If the YAML file is empty, malformed, or missing.

    Returns:
        dict: The parsed content of the YAML file.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        raise CustomExceptionError(e, sys) from e


def create_directories(path_to_directories: list[Path | str], verbose: bool = True):
    """
    Iteratively creates directories from a list of paths.

    Args:
        path_to_directories (list[Path | str]): A list of paths to be created.
        verbose (bool, optional): If True, logs the creation of each directory.
            Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def save_json(path: Path, data: dict):
    """
    Serializes a dictionary into a JSON file.

    Args:
        path (Path): Path where the JSON file will be saved.
        data (dict): The dictionary data to be serialized.
    """
    import json

    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")


def get_size(path: Path) -> str:
    """
    Calculates the file size in Kilobytes.

    Args:
        path (Path): Path to the target file.

    Returns:
        str: Human-readable file size string (e.g., '~ 152 KB').
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
