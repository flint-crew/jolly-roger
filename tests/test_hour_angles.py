"""Some tests around the hour angles"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord

from jolly_roger.hour_angles import get_location, make_hour_angles


def test_askap_position() -> None:
    """Ensure that the EarthLocation for ASKAP is correctly formed"""
    return


def test_get_location_averages_positions() -> None:
    """Averaged geocentric location from antenna positions"""
    positions = np.array(
        [
            [-2556146.7, 5097426.7, -2848333.2],
            [-2556150.0, 5097430.0, -2848330.0],
        ]
    )

    location = get_location(positions)

    assert isinstance(location, EarthLocation)
    assert np.isfinite(location.x.value)


def test_make_hour_angles() -> None:
    """Hour angles, elevation and the time_map from raw time samples"""
    times_mjds = np.array([5156647267.6, 5156647277.6, 5156647287.5]) * u.s
    location = EarthLocation.from_geocentric(
        -2556146.7, 5097426.7, -2848333.2, unit="m"
    )
    position = SkyCoord(ra=294.85 * u.deg, dec=-63.71 * u.deg)

    result = make_hour_angles(
        times_mjds=times_mjds, location=location, position=position
    )

    assert result.hour_angle.shape == (3,)
    assert result.elevation.shape == (3,)
    assert result.position is position

    # time_map keeps unique times in first-seen order (input already sorted here)
    assert list(result.time_map.values()) == [0, 1, 2]
    assert result.time_mjds[0].to(u.s).value == 5156647267.6
