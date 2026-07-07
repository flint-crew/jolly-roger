"""Tests to do with the weights, their idenitiication
and modification based on tapered properties"""

from __future__ import annotations

import numpy as np

from jolly_roger.weights import (
    calculate_scaling_from_taper,
    scale_multiple_weights,
    scale_weights,
)


def test_scale_multiple_weights() -> None:
    """Provided a collection of column and WEIGHT mappings, scale each of them
    and return the result. This is essentially a super set of the tests below"""

    weights_dict = {}
    for column in ("WEIGHTS", "WEIGHT_SPECTRUM"):
        weight_spectrum = np.zeros((256, 100))
        for i in range(256):
            weight_spectrum[i, :] = i + 1

        weights_dict[column] = weight_spectrum

    assert len(weights_dict) == 2

    taper = np.ones((256, 100))
    scaled_weights_dict = scale_multiple_weights(taper=taper, weights=weights_dict)

    for column, weights in weights_dict.items():
        assert np.all(weights == scaled_weights_dict[column])

    taper_1 = taper.copy()
    taper_1[100, 50:] = 0
    scaled_weights_dict = scale_multiple_weights(taper=taper_1, weights=weights_dict)

    for _column, scaled_weights in scaled_weights_dict.items():
        assert np.all(scaled_weights[100] == 202)
        for i in range(100):
            assert np.all(scaled_weights[i] == i + 1)

        for i in range(101, 256):
            assert np.all(scaled_weights[i] == i + 1)


def test_scale_weights_with_weight_spectrum() -> None:
    """Attempt to scale a WEIGHT_SPECTRUM based on the taper. The WEIGHT_SPECTRUM is
    of a shape (row, SPECTRUM)"""

    weight_spectrum = np.zeros((256, 100))
    for i in range(256):
        weight_spectrum[i, :] = i + 1

    taper = np.ones((256, 100))
    scaled_weights = scale_weights(taper=taper, weights=weight_spectrum)
    assert np.all(scaled_weights == weight_spectrum)

    taper_1 = taper.copy()
    taper_1[100, 50:] = 0
    scaled_weights = scale_weights(taper=taper_1, weights=weight_spectrum)

    assert np.all(scaled_weights[100] == 202)
    for i in range(100):
        assert np.all(scaled_weights[i] == i + 1)

    for i in range(101, 256):
        assert np.all(scaled_weights[i] == i + 1)


def test_scale_weights_with_single_value_weight() -> None:
    """Ensure that the weights are appropriately scaled based on the aggressiveness of the taper.
    In this case the weights are not a weight spectrum, but the less informative WEIGHT where all channels
    are weighted equally.
    """

    weights = np.zeros(256) + 50.0
    taper = np.ones((256, 100))
    scaled_weights = scale_weights(taper=taper, weights=weights)

    # Since no nulling the weights should remain the same
    assert np.all(scaled_weights == weights)

    taper_1 = taper.copy()
    taper_1[100, 50:] = 0
    scaled_weights = scale_weights(taper=taper_1, weights=weights)

    assert scaled_weights[100] == 100
    assert np.all(scaled_weights[:100] == 50)
    assert np.all(scaled_weights[101:] == 50)


def test_calculate_scaling_from_taper() -> None:
    """Deriving the scale from the tapering to apply to weights"""

    all_ones = np.ones((256, 100))
    scale = calculate_scaling_from_taper(taper=all_ones)

    assert scale.ndim == 1
    assert np.all(scale == 1)

    example = all_ones.copy()
    example[100, 50:] = 0

    scale = calculate_scaling_from_taper(taper=example)
    assert scale.ndim == 1
    assert scale[100] == 2
    assert np.all(scale[:100] == 1)
    assert np.all(scale[101:] == 1)

    example2 = all_ones.copy()
    example2[100, :] = 0

    scale = calculate_scaling_from_taper(taper=example2)
    assert scale.ndim == 1
    assert scale[100] == 100
    assert np.all(scale[:100] == 1)
    assert np.all(scale[101:] == 1)
