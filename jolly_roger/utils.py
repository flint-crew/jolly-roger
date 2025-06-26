"""Small helper utility functions"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

from jolly_roger import __version__
from jolly_roger.logging import logger


def log_jolly_roger_version() -> None:
    """Write out a jolly roger header to output"""
    logger.info("Jolly-Roger: Flagging and Tapering")
    logger.info(f"Version: {__version__}")


def log_dataclass_attributes(to_log: Any, class_name: str | None = None) -> None:
    """Log the attributes and values from an Dataclass"""
    if not is_dataclass(to_log):
        return

    if class_name:
        logger.info(f"Settings for {class_name}")

    for attribute in to_log.__annotations__:
        value = to_log.__dict__[attribute]
        logger.info(f"{attribute:<30} = {value}")
