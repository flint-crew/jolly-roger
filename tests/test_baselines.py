"""Tests around the baseline functions"""

from __future__ import annotations

import astropy.units as u
import pytest

from jolly_roger.baselines import beam_fraction_to_radius


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
        with pytest.raises(ValueError, match="needs to be between"):
            beam_fraction_to_radius(fraction=fraction, field_of_view=1 * u.deg)
