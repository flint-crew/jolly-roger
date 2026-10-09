"""Tests around the baseline functions"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c as speed_of_light
from casacore.tables import table

from jolly_roger.baselines import (
    beam_fraction_to_radius,
    get_baselines,
    get_nominal_fov,
    get_open_ms_tables,
    get_phase_dir,
)


def test_beam_fraction_to_radius() -> None:
    """Converts the FWHM FoV to a radial distance"""
    fraction = 0.5
    field_of_view = 2 * u.deg

    radial_fov = beam_fraction_to_radius(fraction=fraction, field_of_view=field_of_view)
    assert isinstance(radial_fov, u.Quantity)
    assert field_of_view / 2 == radial_fov


def test_beam_fraction_to_radius_out_of_bounds() -> None:
    """Makes sure that a fraction that is not between 0 to 1 is captured"""

    for fraction in (-1, -0.5, 0, 1, 1.5, 2):
        with pytest.raises(ValueError, match="needs to be in the range"):
            beam_fraction_to_radius(fraction=fraction, field_of_view=1 * u.deg)


def test_get_baselines() -> None:
    """Baseline vectors and maps for the upper triangle of antenna pairs"""
    ant_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    baselines = get_baselines(ant_xyz=ant_xyz)

    np.testing.assert_array_equal(baselines.b_idx, [[0, 1], [0, 2], [1, 2]])
    np.testing.assert_array_equal(
        baselines.b_xyz.value,
        [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [1.0, -2.0, 0.0]],
    )
    assert baselines.b_map == {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    assert baselines.ms_path is None


def test_get_baselines_reversed() -> None:
    """Reversing swaps the antenna ordering in each pair"""
    ant_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    baselines = get_baselines(ant_xyz=ant_xyz, reverse_baselines=True)

    np.testing.assert_array_equal(baselines.b_idx, [[1, 0], [2, 0], [2, 1]])
    assert baselines.b_map == {(1, 0): 0, (2, 0): 1, (2, 1): 2}


def test_get_phase_dir_single_field() -> None:
    """A single FIELD_ID yields the phase direction of that field"""
    field_id = np.zeros(5, dtype=int)
    phase_dir = np.array([[[1.0, -0.5]]])  # (field, poly, [ra, dec]) in rad

    position = get_phase_dir(field_id=field_id, phase_dir=phase_dir)

    assert np.isclose(position.ra.rad, 1.0)
    assert np.isclose(position.dec.rad, -0.5)


def test_get_phase_dir_multi_field_raises() -> None:
    """More than one FIELD_ID is rejected (the >1 guard)"""
    field_id = np.array([0, 1, 0])
    phase_dir = np.array([[[1.0, -0.5]], [[2.0, 0.5]]])

    with pytest.raises(ValueError, match="single FIELD_ID"):
        get_phase_dir(field_id=field_id, phase_dir=phase_dir)


def test_get_nominal_fov() -> None:
    """FoV follows 1.02 * lambda_max / dish at the lowest frequency"""
    chan_freq = np.array([1.0e9, 1.1e9])
    dish_diameter = np.array([12.0, 12.0])

    fov = get_nominal_fov(chan_freq=chan_freq, dish_diameter=dish_diameter)

    expected = (
        (1.02 * (speed_of_light / (1.0e9 * u.Hz)) / (12.0 * u.m)).decompose().value
    )
    assert np.isclose(fov.to("rad").value, expected)


def test_get_nominal_fov_zero_diameter() -> None:
    """A dish diameter of zero has an unbounded field of view, without dividing by zero"""
    fov = get_nominal_fov(
        chan_freq=np.linspace(800e6, 1100e6, 8), dish_diameter=np.zeros(4)
    )
    assert np.isinf(fov.value)


def test_get_open_ms_tables_dish_diameter_override(ms_example: Path) -> None:
    """The dish diameter of the measurement set may be overridden"""
    with table(str(ms_example / "SPECTRAL_WINDOW"), ack=False) as tab:
        lowest_freq = np.min(tab.getcol("CHAN_FREQ")) * u.Hz

    with get_open_ms_tables(ms_path=ms_example) as from_ms:
        ms_fov = from_ms.nominal_fov
    with get_open_ms_tables(ms_path=ms_example, dish_diameter_m=6.0) as overridden:
        fov = overridden.nominal_fov

    expected = (1.02 * (speed_of_light / lowest_freq) / (6.0 * u.m)).decompose() * u.rad
    assert fov.to(u.rad).value == pytest.approx(expected.value)
    # ASKAP dishes are 12 m, so halving the diameter doubles the field of view
    assert fov.to(u.rad).value == pytest.approx(2 * ms_fov.to(u.rad).value)
