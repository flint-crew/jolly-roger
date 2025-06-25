"""Tests to ensure correctness of the
wrapping utilities"""

from __future__ import annotations

import numpy as np

from jolly_roger.wrap import calculate_nyquist_zone, symmetric_domain_wrap


def test_symmetric_domain_wrap() -> None:
    """Ensure mapping values to a periodic domain works"""

    values = np.linspace(10, 40, 10)

    wrapped_values = symmetric_domain_wrap(values=values, upper_limit=60)
    assert np.all(np.isclose(values, wrapped_values))

    values = np.linspace(10, 40, 10)

    # Domain size would be 120 values given the upper limit of 60
    wrapped_values = symmetric_domain_wrap(values=values + 120, upper_limit=60)
    assert np.all(np.isclose(values, wrapped_values))

    values = np.linspace(10, 40, 10)

    # Domain size would be 120 values given the upper limit of 60
    wrapped_values = symmetric_domain_wrap(values=values + 2 * 120, upper_limit=60)
    assert np.all(np.isclose(values, wrapped_values))


def test_calculate_nyquist_zone() -> None:
    """Match to the right zone"""
    assert calculate_nyquist_zone(values=30, upper_limit=60) == 1
    assert calculate_nyquist_zone(values=90, upper_limit=60) == 2
    assert calculate_nyquist_zone(values=-30, upper_limit=60) == 1
    assert calculate_nyquist_zone(values=-90, upper_limit=60) == 2

    assert np.all(
        calculate_nyquist_zone(values=np.array([-90, -30, 0, 30, 90]), upper_limit=60)
        == np.array([2, 1, 1, 1, 2])
    )
