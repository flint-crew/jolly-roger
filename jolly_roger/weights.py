"""Utility functions to help direct the rescaling of weight like
columns"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from casacore.tables import table
from numpy.typing import NDArray

from jolly_roger.logging import logger

KNOWN_WEIGHTS = ("WEIGHT", "WEIGHT_SPECTRUM")


def _get_column_names(ms_path: Path) -> Sequence[str]:
    """A dumb helper to avoid duplication"""
    assert ms_path.exists(), f"MS file {ms_path} does not exist"

    with table(str(ms_path), ack=False, readonly=True) as tab:
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


def select_weight_column(ms_path: Path, weight_column: str | None = None) -> str | None:
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


def calculate_scaling_from_taper(taper: NDArray[np.floating]) -> NDArray[np.floating]:
    """The taper applied in delay space is essentially a Notch filter, which
    has been implemented to have a smooth Gaussian roll off. This will calculate
    the ratio of modified data and return an appropriate scaling term to indicate
    the amount of information nulled away.

    The returned scaling is a multimplicative term that should be applied to the weight column.
    The more data that is nulled the larger this term becomes.

    Args:
        taper (NDArray[np.floating]): The taper that will be applied

    Returns:
        NDArray[np.floating]: The scaling terms to adjust weights by. The more data that is nulled the higher the returned value becomes
    """
    # The aper can be (rows, channels, pols), where pols is really length 1 but is reshaped
    # accordingly to t the recorded MS data column polarisations
    taper = np.squeeze(taper)

    # Basic checks around the taper
    assert taper.ndim == 2, f"Expectede a 2D array, got {taper.shape=}"
    assert np.min(taper) >= 0.0, "Taper appears to be less than zero"
    assert np.max(taper) <= 1.0, "Taper appears to be larger than one"

    # The taper should be of shape (row, signal), where row is simply the
    # index into the MS data table
    signal_length = taper.shape[1]
    signal = np.sum(taper, axis=1)
    signal[signal == 0] += 1  # Avoid divide by zero
    return signal_length / signal


def scale_weights(
    taper: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Accept the taper array and associated set of weights, and appropriately
    scale the weights based on the amount of data nulled. The weights here could
    take a couple of forms:

    - WEIGHT: a single value for a collection of rows
    - WEIGHT_SPECTRUM: A spectrum of weight values (e.g. channel weights) across a collection of rows

    The taper is assumed to be the Notch filter, with a smooth roll off, and in the 0 to 1 range. Where
    data is nulled (closer to 0) the weights will increase. Since the taper is intended to be applied
    in delay space and then subsequently inverted to visibilities, the weights are simply scaled in a
    pure multiplicative sense. In either the WEIGHT or WEIGHT_SPECTRUM scale a single scalar is used to
    scale a single row.

    Args:
        taper (NDArray[np.floating]): The two-dimensional taper that will be applied to data
        weights (NDArray[np.floating]): The extract weights column

    Raises:
        ValueError: Raised when the supplied weight have a rank that is neither 1 or 2

    Returns:
        NDArray[np.floating]: Scaled weights that should be inserted to the measuremenset set
    """

    scale = calculate_scaling_from_taper(taper=taper)

    scaled_weights: None | NDArray[np.floating] = None
    # Appropriately handle potential broadcasting issues, e.g. WEIGHT vs WEIGHT_SPECTRUM
    if weights.ndim == 1:
        # No broadcasting is needed
        scaled_weights = weights * scale
    elif weights.ndim == 2:
        # The shape should be [row, channel], and the scaling will be per-channel
        scaled_weights = weights * scale[:, None]
    else:
        msg = f"Can only handle 1D or 2D weights, got {weights.shape=}"
        raise ValueError(msg)

    assert scaled_weights is not None, "Scaled weights appears unset"
    return scaled_weights
