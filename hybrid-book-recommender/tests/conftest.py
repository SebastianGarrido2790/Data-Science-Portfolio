import pytest

from src.utils.logger import get_logger


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """
    Ensures that the FileHandler is attached to the root logger during tests.
    Pytest's default logging configuration might capture or remove custom handlers,
    so we re-initialize our logger to ensure logs are written to running_logs.log.
    """
    # Simply calling get_logger will execute the logic to attach handlers if missing
    get_logger()
