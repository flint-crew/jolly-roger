"""Utility functions to help direct the rescaling of weight like
columns"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


def find_weight_columns(ms_path: Path) -> Sequence[str] | None:
    """Examine the column names in the MS in order to find
    WEIGHT-like columns. The column names are drawn from
    a list of recognised column names that may contain WEUGHTS.

    Args:
        ms_path (Path): Path to the measurement set

    Returns:
        Sequence[str] | None: The names of the WEIGHT-like column. If not found None is returned.
    """
    logger.info("Searching for WEIGHT-like column")

    columns = _get_column_names(ms_path)
    weight_columns = [col for col in KNOWN_WEIGHTS if col in columns]

    if len(columns) == 0:
        logger.info("No WEIGHT-like columns were found.")
        return None

    logger.info(f"Found WEIGHT-like column {weight_columns=}")
    return weight_columns


def select_weight_columns(
    ms_path: Path, weight_column: str | None = None
) -> Sequence[str] | None:
    """Establishes the set ofWEIGHT-like column from provided options.

    Known WEIGHT-like column names will be examined to see if they are present in
    the MS. Multiple columns may be returned in more than one are found. Otherwise
    a single ``weight_column`` may be provided, and if found to be valid, will
    overwrite the recognised columns.

    Args:
        ms_path (Path): The measurement set to consider
        weight_column (str | None, optional): The specified weight column to use. Defaults to None.

    Raises:
        ValueError: Raised if ``weight_column`` is provided but does not exist

    Returns:
        Sequence[str] | None: Names of the WEIGHT-like column if found or set. If not found None is returned.
    """

    if weight_column:
        columns = _get_column_names(ms_path=ms_path)

        # If the user requested a column that does not exist
        # we hit them with an error, mate
        if weight_column not in columns:
            msg = f"Specified {weight_column=} not found in {columns=}"
            raise ValueError(msg)

        logger.info(f"Using specified {weight_column=} as WEIGHT-like column")
        return [weight_column]

    return find_weight_columns(ms_path=ms_path)


def calculate_scaling_from_taper(
    taper: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """The taper applied in delay space is essentially a Notch filter, which
    has been implemented to have a smooth Gaussian roll off. This will calculate
    the ratio of modified data and return an appropriate scaling term to indicate
    the amount of information nulled away.

    The returned scaling is a multimplicative term that should be applied to the weight column.
    The more data that is nulled the larger this term becomes.

    Args:
        taper (NDArray[np.floating[Any]]): The taper that will be applied

    Returns:
        NDArray[np.floating[Any]]: The scaling terms to adjust weights by. The more data that is nulled the higher the returned value becomes
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
    taper: NDArray[np.floating[Any]],
    weights: NDArray[np.floating[Any]],
    taper_is_scale: bool = False,
) -> NDArray[np.floating[Any]]:
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
        taper (NDArray[np.floating[Any]]): The two-dimensional taper that will be applied to data
        weights (NDArray[np.floating[Any]]): The extract weights column
        taper_is_scale (bool, optional): Indicates whether the input ``taper`` has already been t ransformed to a scaling term. Defaults to False.

    Raises:
        ValueError: Raised when the supplied weight have a rank that is neither 1 or 2

    Returns:
        NDArray[np.floating[Any]]: Scaled weights that should be inserted to the measuremenset set
    """
    # The scale could be already derieved should multi-ple columns need scaling
    scale = calculate_scaling_from_taper(taper=taper) if not taper_is_scale else taper

    scaled_weights: None | NDArray[np.floating[Any]] = None
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


def scale_multiple_weights(
    taper: NDArray[np.floating[Any]],
    weights: dict[str, NDArray[np.floating[Any]]],
) -> dict[str, NDArray[np.floating[Any]]]:
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
        taper (NDArray[np.floating[Any]]): The two-dimensional taper that will be applied to data
        weights (dict[str, NDArray[np.floating[Any]]]): The extract weights column. For each key (representing the column name) scakle the mapped data (the weights)

    Returns:
        dict[str, NDArray[np.floating[Any]]]: Scaled weights that should be inserted to the measuremenset set. The output shape will represent the scaled input ``weights``
    """

    assert isinstance(weights, dict), (
        f"Expected weights to be a dictionary, got {type(weights)}"
    )

    scale = calculate_scaling_from_taper(taper=taper)

    return {
        k: scale_weights(taper=scale, weights=weights[k], taper_is_scale=True)
        for k in weights
    }
