"""Some tests around the hour angles"""

from __future__ import annotations

from astropy.coordinates import EarthLocation


def test_askap_position() -> None:
    """Ensure that the EarthLocation for ASKAP is correctly formed"""
    from jolly_roger.hour_angles import ASKAP

    askap_astropy = EarthLocation.of_site("ASKAP")

    assert ASKAP.x.value == askap_astropy.x.value
    assert ASKAP.y.value == askap_astropy.y.value
    assert ASKAP.z.value == askap_astropy.z.value
