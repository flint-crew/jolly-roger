"""Tests around the tapering construction functions
for the tukey taper"""

from __future__ import annotations

import numpy as np
import pytest

from jolly_roger.tapering.tukey import (
    InconsistentLengthError,
    get_2d_taper,
    make_inputs_consistent,
    verify_overlap_widths,
)


def test_get_2d_taper() -> None:
    """Evaluate the construction of the tukey taper. These are
    basic checks that are mostly testing for user error."""

    x = np.arange(80) - 40

    taper = get_2d_taper(x=x, outer_width=6, tukey_width=2, tukey_offset=None)

    assert taper.ndim == 2
    assert taper.shape[0] == len(x)
    assert taper.shape[1] == 1

    taper = get_2d_taper(
        x=x, outer_width=np.arange(10) + 4, tukey_width=1, tukey_offset=None
    )

    assert taper.ndim == 2
    assert taper.shape[0] == len(x)
    assert taper.shape[1] == 10


def test_get_2d_taper_wraps_at_half_period() -> None:
    """A notch centred near the top of an even-length Fourier axis must wrap
    onto the bottom of the axis with the true period (N * dx), not 2 * max(x)"""
    n = 8
    x = np.fft.fftshift(np.fft.fftfreq(n, d=1 / n))  # -4 ... 3, period 8

    taper = get_2d_taper(
        x=x, outer_width=1.2, tukey_width=0.2, tukey_offset=np.array([3.5])
    )[:, 0]

    # x = 3 and x = -4 are both 0.5 from the centre once wrapped, so fully nulled
    assert taper[x == 3][0] == 0.0
    assert taper[x == -4][0] == 0.0
    # x = 1 is 2.5 away, beyond the notch
    assert taper[x == 1][0] == 1.0


def test_verify_overlap_widths_with_reset() -> None:
    """See if the resetting of outer_width in instances
    where it is too small for the tukey width is correctly handled.
    Here the resetting is invoked"""

    outer = np.array([10] * 10)
    tukey = np.array([20] * 10)

    outer_checked, tukey_checked = verify_overlap_widths(
        outer_width=outer, tukey_width=tukey
    )
    assert np.all(tukey == outer_checked)
    assert np.all(tukey == tukey_checked)


def test_verify_overlap_widths() -> None:
    """See if the resetting of outer_width in instances
    where it is too small for the tukey width is correctly handled"""

    outer = np.array([100] * 10)
    tukey = np.array([20] * 10)

    outer_checked, tukey_checked = verify_overlap_widths(
        outer_width=outer, tukey_width=tukey
    )
    assert np.all(outer == outer_checked)
    assert np.all(tukey == tukey_checked)


def test_make_outputs_consistent() -> None:
    """floats go in, arrays come out"""

    outer, tukey, offset, max_dim = make_inputs_consistent(
        outer_width=20, tukey_width=10, tukey_offset=None
    )
    assert isinstance(outer, np.ndarray)
    assert isinstance(tukey, np.ndarray)
    assert outer.ndim == 1
    assert tukey.ndim == 1
    assert outer.shape == tukey.shape
    assert outer.shape == offset.shape
    assert max_dim == 1
    assert np.all(offset == 0)


def test_make_outputs_consistent_2() -> None:
    """floats go in, arrays come out"""

    outer, tukey, offset, max_dim = make_inputs_consistent(
        outer_width=np.arange(10), tukey_width=10, tukey_offset=None
    )
    assert isinstance(outer, np.ndarray)
    assert isinstance(tukey, np.ndarray)
    assert outer.ndim == 1
    assert tukey.ndim == 1
    assert outer.shape == tukey.shape
    assert max_dim == 10
    assert np.all(offset == 0)


def test_make_outputs_consistent_raise_error() -> None:
    """Appropriate error raised when single broadcastable shape
    is not possible"""

    with pytest.raises(InconsistentLengthError):
        _ = make_inputs_consistent(
            outer_width=np.arange(2), tukey_width=np.arange(5), tukey_offset=None
        )

    with pytest.raises(InconsistentLengthError):
        _ = make_inputs_consistent(
            outer_width=np.arange(5),
            tukey_width=np.arange(5),
            tukey_offset=np.arange(1200),
        )
