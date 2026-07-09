"""Sanity checks against a mini MS"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import EarthLocation
from casacore.tables import table

from jolly_roger.baselines import _get_phase_dir_for_field_id, get_baselines_from_ms
from jolly_roger.hour_angles import get_location_from_ms, make_hour_angles_for_ms
from jolly_roger.uvws import get_object_delay_for_ms


def test_baselines(
    ms_example: Path,
    ms_ant_xyz: np.ndarray,
    ms_b_idx: np.ndarray,
    ms_b_xyz: np.ndarray,
    ms_b_map: dict[Any, Any],
) -> None:
    baselines = get_baselines_from_ms(ms_path=ms_example)
    np.testing.assert_array_equal(baselines.ant_xyz.value, ms_ant_xyz)
    np.testing.assert_array_equal(baselines.b_idx, ms_b_idx)
    np.testing.assert_array_equal(baselines.b_xyz.value, ms_b_xyz)
    for key, val in baselines.b_map.items():
        assert ms_b_map[key] == val


def test_location(
    ms_example: Path,
    ms_location: EarthLocation,
) -> None:
    location = get_location_from_ms(ms_path=ms_example)
    assert np.isclose(location.x, ms_location.x)
    assert np.isclose(location.y, ms_location.y)
    assert np.isclose(location.z, ms_location.z)


def test_hour_angles(
    ms_example: Path,
    ms_location: EarthLocation,
) -> None:
    hour_angles = make_hour_angles_for_ms(ms_path=ms_example, position="PKSB1934-638")
    np.testing.assert_allclose(
        hour_angles.hour_angle.deg, np.array([174.61841401, 174.65999955, 174.70158509])
    )
    np.testing.assert_allclose(
        hour_angles.time_mjds.value,
        np.array([5.15664727e09, 5.15664728e09, 5.15664729e09]),
    )

    assert np.isclose(hour_angles.location.x, ms_location.x)
    assert np.isclose(hour_angles.location.y, ms_location.y)
    assert np.isclose(hour_angles.location.z, ms_location.z)

    assert hour_angles.position.ra.deg == 294.854277
    assert hour_angles.position.dec.deg == -63.712673

    np.testing.assert_allclose(
        hour_angles.elevation.deg, np.array([0.47442616, 0.47274064, 0.47106702])
    )

    np.testing.assert_allclose(
        hour_angles.time.mjd,
        np.array([59683.41744939, 59683.41756459, 59683.41767979]),
    )

    for key, val in hour_angles.time_map.items():
        if val == 0:
            assert key.value == 5156647267.627391  # type: ignore[attr-defined]
        if val == 1:
            assert key.value == 5156647277.58067  # type: ignore[attr-defined]
        if val == 2:
            assert key.value == 5156647287.533951  # type: ignore[attr-defined]


def test_get_object_delay_for_ms(
    ms_example: Path,
    ms_b_map: dict[Any, Any],
    ms_w_delays: np.ndarray,
) -> None:
    with (
        table(ms_example.as_posix(), ack=False) as ms_table,
        table((ms_example / "FIELD").as_posix(), ack=False) as field_table,
    ):
        phase_dir = _get_phase_dir_for_field_id(
            ms_table,
            field_table,
        )

    w_delays = get_object_delay_for_ms(
        ms_path=ms_example, phase_dir=phase_dir, object_name="PKSB1934-638"
    )
    assert len(w_delays) == 1
    w_delay = w_delays[0]

    assert w_delay.object_name == "PKSB1934-638"

    for key, val in w_delay.b_map.items():
        assert ms_b_map[key] == val

    for key, val in w_delay.time_map.items():  # type: ignore[assignment]
        if val == 0:
            assert key.value == 5156647267.627391  # type: ignore[attr-defined]
        if val == 1:
            assert key.value == 5156647277.58067  # type: ignore[attr-defined]
        if val == 2:
            assert key.value == 5156647287.533951  # type: ignore[attr-defined]

    np.testing.assert_allclose(
        w_delay.elevation.deg, np.array([0.47442616, 0.47274064, 0.47106702])
    )

    assert w_delay.guard_region is None

    np.testing.assert_allclose(w_delay.w_delays.value, ms_w_delays)
