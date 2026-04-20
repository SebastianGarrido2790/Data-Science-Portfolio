"""
Centralized constants for the project.

This module stores immutable values like file paths to ensuring
consistency across the entire MLOps pipeline.
"""

from pathlib import Path

# Paths to the primary configuration files
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")
SCHEMA_FILE_PATH = Path("config/schema.yaml")
