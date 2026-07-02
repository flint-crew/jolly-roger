"""Helper functions and utilities to assist with the management of sinc responses in delay space"""

from __future__ import annotations

from typing import Any

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from jolly_roger.logging import logger

"""This section deals with attempting to anaylitically construct properties of an
idealised sinc response from an impulse. The limits of normalised sinc response:

>>> sin(pi * x) / (pi * x)

spans +/- 1. If we can constructed the expected range in delay space we ought
to be able to automatically set a taper width and predict the location of the
N'th sidelobe.
"""
# TODO: The above needs some proper math refs to support. Got here through
# through some reading and toying.


def calculate_expected_sinc_width(
    freqs: NDArray[np.floating[Any]] | u.Quantity,
) -> u.Quantity:
    """The expected width of a sinc response in delay space given a set of frequencies,

    The return result is +/- from 0. In otherwords the domain of the normalised
    sinc function is 2 * result. The width of the sinc response is determined by the
    bandwidth of the frequencies.

    Args:
        freqs (NDArray[np.floating[Any]] | u.Quantity): The sampled frequencies. If units are not specified MHz are assumed.

    Returns:
        u.Quantity: The expected width of the sinc response
    """
    if not isinstance(freqs, u.Quantity):
        freqs = freqs * u.MHz  # Assume MHz if no units are provided

    bandwidth = np.max(freqs) - np.min(freqs)  # Calculate bandwidth in Hz
    sinc_width = (
        (1 / bandwidth).decompose().to("s")
    )  # The width of the sinc response in seconds

    logger.info(f"Calculated expected sinc width: {sinc_width=} for {bandwidth=}")

    return sinc_width


def get_delay_of_nth_sidelobe(
    n: int, sinc_width: u.Quantity | np.floating[Any]
) -> u.Quantity:
    """Calculate the expected delay of the N'th sidelobe of a sinc response given the width of the sinc response.

    The main response is located at delay 0 seconds and ends at +/1 sinc_width. The peak position of the N'th sidelobe is:

    >>> (n + 0.5) * sinc_width

    For the special case of n=0, 0 seconds is returned, which is the location of the main response.

    Args:
        n (int): The order of the sidelobe (1 for first sidelobe, 2 for second, etc.)
        sinc_width (u.Quantity | np.floating[Any]): The width of the sinc response in seconds. If no units are attached seconds are assumed.

    Returns:
        u.Quantity: The expected delay of the N'th sidelobe in seconds.
    """
    if not isinstance(sinc_width, u.Quantity):
        sinc_width *= u.s

    if n == 0:
        return 0 * u.s

    return sinc_width * (n + 0.5)
