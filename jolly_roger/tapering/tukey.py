"""Functions and helpers around constructing the tukey tapering function to be used
when applying a filter in delay space."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from jolly_roger.logging import logger
from jolly_roger.wrap import symmetric_domain_wrap


class InconsistentLengthError(Exception):
    pass


def make_inputs_consistent(
    outer_width: float | NDArray[np.floating[Any]],
    tukey_width: float | NDArray[np.floating[Any]],
    tukey_offset: NDArray[np.floating[Any]] | None = None,
) -> tuple[
    NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
]:
    """Attempt to form consistent dimensions between the parametisation of the tukey
    taper so that they can be broadcastable. The returned taper will be of shape (NFreqs, NTimesteps).
    If all these inputs are floats, then the taper can be applied equally to all rimsteps.
    Otherwise if a single input varies across timesteps, then others need to be changed to
    be broadcastable into that shape.

    Args:
        outer_width (float | NDArray[np.floating[Any]]): The total size of the taper (inner region + the tapering down region)
        tukey_width (float | NDArray[np.floating[Any]]): The size of the tapering down region
        tukey_offset (NDArray[np.floating[Any]] | None, optional): Whether to offset the origin of the taper. Defaults to None.

    Raises:
        ValueError: e

    Returns:
        tuple[ NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]], int ]: Arrays that are consistently shaped for broadcasting
    """

    outer_width_array: NDArray[np.floating[Any]] = (
        np.array([outer_width]) if np.isscalar(outer_width) else outer_width
    )
    tukey_width_array: NDArray[np.floating[Any]] = (
        np.array([tukey_width]) if np.isscalar(tukey_width) else tukey_width
    )
    tukey_offset_array: NDArray[np.floating[Any]] = (
        np.array([0.0]) if tukey_offset is None else tukey_offset
    )

    # A length of 1 is OK as it is going to be repeated on return
    sizes = (outer_width_array.size, tukey_width_array.size, tukey_offset_array.size)
    unique_lengths = set(sizes) - {
        1,
    }
    if len(unique_lengths) > 1:
        msg = f"Unable to support {sizes=}. Ensure all same length or of size 1"
        raise InconsistentLengthError(msg)

    max_dim = max(sizes)
    return (
        outer_width_array
        if outer_width_array.size == max_dim
        else np.repeat(outer_width_array, max_dim),
        tukey_width_array
        if tukey_width_array.size == max_dim
        else np.repeat(tukey_width_array, max_dim),
        tukey_offset_array
        if tukey_offset_array.size == max_dim
        else np.repeat(tukey_offset_array, max_dim),
        max_dim,
    )


def verify_overlap_widths(
    outer_width: NDArray[np.floating[Any]], tukey_width: NDArray[np.floating[Any]]
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Ehsure that ``outer_width`` is appropriately sized for supplied ``tukey_width``.
    The parameterisation of the tukey taper defines the width of the taper to include
    the transition component of the function.

    If the total size (set by ``outer_width``) is not large enough to encompass the
    ``tukey_width`` (i.e. the transition component of the function) then it will be
    resized.

    Args:
        outer_width (NDArray[np.floating[Any]]): The total size of the null and transition
        tukey_width (NDArray[np.floating[Any]]): Size of the transition

    Returns:
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]: Resized outer zone
    """

    mask = outer_width - tukey_width < 0.0

    if np.any(mask):
        logger.warning(
            f"{np.sum(mask)} outer widths to small to support supplied tukey width, resetting minimum width to support"
        )
        outer_width[mask] = tukey_width[mask]

    return outer_width, tukey_width


def get_2d_taper(
    x: NDArray[np.floating[Any]],
    outer_width: NDArray[np.floating[Any]] | float,
    tukey_width: NDArray[np.floating[Any]] | float,
    tukey_offset: NDArray[np.floating[Any]] | None = None,
) -> NDArray[np.floating[Any]]:
    """Calculate a tapering function across the domain defined by ``x``. The output
    shape of this function will always be two-dimensional ``(x, N)``, where ``N`` is
    either ``1`` or the length of the longest tapering option.

    Internally the width and offset parameters are converted to arrays appropriate for
    broadcasting.

    Unless otherwise set by ``tukey_offset``, the center of the taper is set at
    the origin. This function implements wrapping of around the cyclic boundary of
    ``x``.

    The taper is not two-dimensional in shape. Instead each row is intended to be
    independent from the next, as appropriate for rows in a MS.

    Args:
        x (np.typing.NDArray[np.floating[Any]]): The coordinates to sample across
        outer_width (NDArray[np.floating[Any]] | float): The total size of the taper, including the transition width
        tukey_width (NDArray[np.floating[Any]] | float): The width of the transition region
        tukey_offset (NDArray[np.floating[Any]] | None, optional): A shift of the center of the origin. Defaults to None.

    Returns:
        NDArray[np.floating[Any]]: A two-dimensional array, with first axis being the same length as ``x``.
    """

    outer_width, tukey_width, tukey_offset, _ = make_inputs_consistent(
        outer_width, tukey_width, tukey_offset
    )
    outer_width, tukey_width = verify_overlap_widths(outer_width, tukey_width)

    assert not isinstance(outer_width, float)
    assert not isinstance(tukey_width, float)
    assert tukey_offset is not None

    original_x_local_maximum = np.max(x)
    x_local = x[:, None] - tukey_offset[None, :]

    x_local = symmetric_domain_wrap(
        values=x_local, upper_limit=original_x_local_maximum
    )

    taper = np.ones_like(x_local)

    outer_region = (-outer_width[None, :] < x_local) & (x_local < outer_width[None, :])

    taper[outer_region] = 0.0

    left_idx = (-outer_width[None, :] <= x_local) & (
        x_local <= -outer_width[None, :] + tukey_width[None, :]
    )
    right_idx = (outer_width[None, :] - tukey_width[None, :] <= x_local) & (
        x_local <= outer_width[None, :]
    )

    left_taper = (
        1 - np.cos(np.pi * (x_local + outer_width[None, :]) / tukey_width[None, :])
    ) / 2
    taper[left_idx] = 1 - left_taper[left_idx]

    right_taper = (
        1 - np.cos(np.pi * (outer_width[None, :] - x_local) / tukey_width[None, :])
    ) / 2
    taper[right_idx] = 1 - right_taper[right_idx]

    return taper
