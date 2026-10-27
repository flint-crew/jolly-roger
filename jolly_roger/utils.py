"""Small helper utility functions"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from pydantic import BaseModel

from jolly_roger import __version__
from jolly_roger.logging import logger


def log_jolly_roger_version() -> None:
    """Write out a jolly roger header to output"""
    logger.info("Jolly-Roger: Flagging and Tapering")
    logger.info(f"Version: {__version__}")


def options_items(options: Any) -> list[tuple[str, Any]]:
    """The ``(name, value)`` pairs of a dataclass or pydantic options instance

    Args:
        options (Any): A dataclass or pydantic options instance

    Returns:
        list[tuple[str, Any]]: One pair per field. Empty if ``options`` is neither.
    """
    if isinstance(options, BaseModel):
        return [(name, getattr(options, name)) for name in type(options).model_fields]
    if is_dataclass(options) and not isinstance(options, type):
        return [(field.name, getattr(options, field.name)) for field in fields(options)]
    return []


def log_dataclass_attributes(to_log: Any, class_name: str | None = None) -> None:
    """Log the attributes and values from a dataclass or pydantic options class"""
    items = options_items(to_log)
    if not items:
        return

    if class_name:
        logger.info(f"Settings for {class_name}")

    for attribute, value in items:
        logger.info(f"{attribute:<30} = {value}")
