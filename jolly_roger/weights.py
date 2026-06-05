"""Utility functions to help direct the rescaling of weight like
columns"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from casacore.tables import table

from jolly_roger.logging import logger

KNOWN_WEIGHTS = ("WEIGHT", "WEIGHT_SPECTRUM")


def _get_column_names(ms_path: Path) -> Sequence[str]:
    """A dumb helper to avoid duplication"""
    assert ms_path.exists(), f"MS file {ms_path} does not exist"

    with table(str(ms_path)) as tab:
        columns: Sequence[str] = tab.colnames()

    return columns


def find_weight_column(ms_path: Path) -> str | None:
    """Examine the column names in the MS in order to find
    a WEIGHT-like column. The column names are drawn from
    a list of recognised column names that may contain WEUGHTS.

    Args:
        ms_path (Path): Path to the measurement set

    Returns:
        str | None: The name of the WEIGHT-like column. If not found None is returned.
    """
    logger.info("Searching for WEIGHT-like column")

    weight_column: str | None = None
    columns = _get_column_names(ms_path)

    for col in KNOWN_WEIGHTS:
        if col in columns:
            weight_column = col
            break

    logger.info(f"Found WEIGHT-like column {weight_column=}")
    return weight_column


def set_weight_column(ms_path: Path, weight_column: str | None = None) -> str | None:
    """Establish a WEIGHT-like column from provided options. If ``weight_column``
    is set then verify it exists in the pfovided measurement set. If it is not
    then the columns in the measurement set are examined for a recognised
    WEIGHT-like column.

    Args:
        ms_path (Path): The measurement set to consider
        weight_column (str | None, optional): The specified weight column to use. Defaults to None.

    Raises:
        ValueError: Raised if ``weight_column`` is provided but does not exist

    Returns:
        str | None: Name of the WEIGHT-like column if found or set. If not found None is returned.
    """

    if weight_column:
        columns = _get_column_names(ms_path=ms_path)

        # If the user requested a column that does not exist
        # we hit them with an error, mate
        if weight_column not in columns:
            msg = f"Specified {weight_column=} not found in {columns=}"
            raise ValueError(msg)

        logger.info(f"Using specified {weight_column=} as WEIGHT-like column")
        return weight_column

    return find_weight_column(ms_path=ms_path)
