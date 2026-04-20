"""
Project-wide path constants.

Single source of truth for all configuration file paths, data directories,
and artifact locations. Every pipeline module should import paths from here
instead of hardcoding strings.
"""

from pathlib import Path

# --- Project Root ---
# Automatically finds the top-level directory (the one containing 'src/')
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Configuration & Schema ---
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"
PARAMS_FILE_PATH = CONFIG_DIR / "params.yaml"
SCHEMA_FILE_PATH = CONFIG_DIR / "schema.yaml"

# --- Model & Reports ---
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = REPORTS_DIR / "docs"

# --- Logs ---
LOGS_DIR = PROJECT_ROOT / "logs"

# --- Ensure directories exist ---
directories_to_create = [
    REPORTS_DIR,
    FIGURES_DIR,
    DOCS_DIR,
    LOGS_DIR,
]

for path in directories_to_create:
    path.mkdir(parents=True, exist_ok=True)
