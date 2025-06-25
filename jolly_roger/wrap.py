"""Helper utilities to deal with cyclic boundaries
on data"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def symmetric_domain_wrap(
    values: NDArray[np.floating], upper_limit: float
) -> NDArray[np.floating]:
    """Place a set of values into a cyclic domain that is
    symmetric around zero.

    Args:
        values (NDArray[np.floating]): Values that need to be mapped into the cyclic domain
        upper_limit (float): The upper bound of the symmetric cyclic domain

    Returns:
        NDArray[np.floating]: Values that have been mapped to the -upper_limit to upper_limit domain
    """
    # Calculate an appropriate domain mapping
    # The natural domain is going to be mapped
    # to -pi to pi.
    domain_mapping = np.pi / upper_limit

    real = np.cos(values * domain_mapping)
    imag = np.sin(values * domain_mapping)

    wrapped_values = real + 1j * imag

    return np.angle(wrapped_values) / domain_mapping


def calculate_nyquist_zone(
    values: NDArray[np.floating], upper_limit: float
) -> NDArray[np.int_]:
    """Return the nyquist zone a value is in for a symmetric
    set of bounds around zero

    Args:
        values (NDArray[np.floating]): The values to calculate the zone for
        upper_limit (float): The upper bound to the symmetric domain around zero

    Returns:
        NDArray[np.int_]: The zones values correspond to
    """
    return np.array(
        np.floor((upper_limit + np.abs(values)) / (2.0 * upper_limit)) + 1, dtype=int
    )
