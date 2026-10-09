"""Tests to ensure correctness of the
wrapping utilities"""

from __future__ import annotations

import numpy as np
import pytest

from jolly_roger.wrap import (
    axis_half_period,
    calculate_nyquist_zone,
    symmetric_domain_wrap,
)


def test_symmetric_domain_wrap_with_2d_arrays() -> None:
    """Ensure mapping values to a periodic domain works for arrays"""

    values = np.array([np.linspace(10, 40, 10), np.linspace(10, 40, 10)])
    assert values.shape == (2, 10)

    wrapped_values = symmetric_domain_wrap(values=values, upper_limit=60)
    assert wrapped_values.shape == values.shape
    assert np.all(np.isclose(wrapped_values, values))

    # The addition of 120 is the size of the domain here
    # and they should be wrapped back to the main map
    larger_values = np.array(
        [np.linspace(10, 40, 10), np.linspace(10 + 120, 40 + 120, 10)]
    )
    assert values.shape == (2, 10)

    wrapped_values = symmetric_domain_wrap(values=larger_values, upper_limit=60)
    assert wrapped_values.shape == values.shape
    assert np.all(np.isclose(wrapped_values, values))


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


@pytest.mark.parametrize("n", [64, 63, 288])
def test_axis_half_period(n: int) -> None:
    """The half-period of a shifted Fourier axis is N * dx / 2, which for even N is
    one sample beyond the largest value"""
    spacing = 0.25
    axis = np.fft.fftshift(np.fft.fftfreq(n, d=1 / (n * spacing)))
    assert axis_half_period(axis) == pytest.approx(n * spacing / 2)
    if n % 2 == 0:
        assert axis_half_period(axis) == pytest.approx(np.max(axis) + spacing)
