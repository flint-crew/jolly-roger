"""Some tests around the hour angles"""

from __future__ import annotations


def test_askap_position() -> None:
    """Ensure that the EarthLocation for ASKAP is correctly formed"""
    from jolly_roger.hour_angles import ASKAP

    assert ASKAP.x == -2556146.66356375
    assert ASKAP.y == 5097426.58592797
    assert ASKAP.z == 2848333.08164107
