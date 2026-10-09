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


def test_get_2d_taper_upper_limit() -> None:
    """With an even number of samples the largest value of a shifted Fourier axis
    is one sample short of the half-period. A notch crossing the edge then wraps to
    the wrong place unless the half-period is given."""
    n, spacing = 64, 1.0
    x = np.fft.fftshift(np.fft.fftfreq(n, d=1 / (n * spacing)))  # -32 .. 31
    # A notch centred beyond the edge, at 36, should reach -30 (36 - 64 = -28 +/- 3)
    kwargs = {
        "x": x,
        "outer_width": 3.0,
        "tukey_width": 1.0,
        "tukey_offset": np.array([36.0]),
    }

    wrapped = get_2d_taper(**kwargs, upper_limit=n * spacing / 2)[:, 0]
    assert wrapped[x == -28.0] == 0.0
    assert np.all(wrapped[np.abs(x + 28.0) > 3.0] == 1.0)

    # The default keeps the historical behaviour of wrapping at the largest value
    default = get_2d_taper(**kwargs)[:, 0]
    np.testing.assert_array_equal(
        default, get_2d_taper(**kwargs, upper_limit=float(np.max(x)))[:, 0]
    )
