"""Tests around the UVW delays, sun scales and flagging"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from casacore.tables import table

from jolly_roger.baselines import Baselines, get_baselines_from_ms
from jolly_roger.hour_angles import PositionHourAngles, make_hour_angles_for_ms
from jolly_roger.uvws import (
    SunScale,
    UVWs,
    compute_sun_uv_scales,
    compute_uvw_flags,
    uvw_flagger,
    xyz_to_uvw,
)


def _one_baseline_uvws(uv_dist_m: float, elevation_deg: float) -> UVWs:
    """A single baseline at a given (u,v)-distance and elevation, one time step"""
    uvws = np.array([[[uv_dist_m]], [[0.0]], [[0.0]]]) * u.m  # [coord, baseline, time]
    baselines = Baselines(
        ant_xyz=np.zeros((2, 3)) * u.m,
        b_xyz=np.zeros((1, 3)) * u.m,
        b_idx=np.array([[0, 1]]),
        b_map={(0, 1): 0},
    )
    hour_angles = PositionHourAngles(
        hour_angle=np.array([0.0]) * u.rad,
        time_mjds=np.array([0.0]) * u.s,
        location=EarthLocation.from_geocentric(0, 0, 0, unit="m"),
        position=SkyCoord(0 * u.deg, 0 * u.deg),
        elevation=np.array([elevation_deg]) * u.deg,
        time=Time([59000.0], format="mjd", scale="utc"),
        time_map={},
    )
    return UVWs(uvws=uvws, hour_angles=hour_angles, baselines=baselines)


def test_compute_uvw_flags_short_baseline_flagged() -> None:
    """A short baseline within the horizon is flagged"""
    sun_scale = SunScale(
        min_scale_chan_lambda=np.array([100.0]) * u.m,
        chan_lambda=np.array([1.0]) * u.m,
        min_scale_deg=0.075,
    )
    uvws = _one_baseline_uvws(uv_dist_m=10.0, elevation_deg=45.0)

    result = compute_uvw_flags(computed_uvws=uvws, sun_scale=sun_scale)

    assert (0, 1) in result.flags
    assert result.flags[(0, 1)].all()


def test_compute_uvw_flags_long_baseline_untouched() -> None:
    """A long baseline is not sensitive to the Sun, so no flags"""
    sun_scale = SunScale(
        min_scale_chan_lambda=np.array([100.0]) * u.m,
        chan_lambda=np.array([1.0]) * u.m,
        min_scale_deg=0.075,
    )
    uvws = _one_baseline_uvws(uv_dist_m=1000.0, elevation_deg=45.0)

    result = compute_uvw_flags(computed_uvws=uvws, sun_scale=sun_scale)

    assert result.flags == {}


def test_compute_uvw_flags_below_horizon_untouched() -> None:
    """Below the horizon limit nothing is flagged even for short baselines"""
    sun_scale = SunScale(
        min_scale_chan_lambda=np.array([100.0]) * u.m,
        chan_lambda=np.array([1.0]) * u.m,
        min_scale_deg=0.075,
    )
    uvws = _one_baseline_uvws(uv_dist_m=10.0, elevation_deg=-30.0)

    result = compute_uvw_flags(computed_uvws=uvws, sun_scale=sun_scale)

    assert result.flags == {}


def test_compute_sun_uv_scales() -> None:
    """(u,v)-distance sensitive to an angular scale scales as lambda / theta"""
    chan_freqs = np.array([1.0e9]) * u.Hz
    min_scale = 0.1 * u.rad

    sun_scale = compute_sun_uv_scales(chan_freqs=chan_freqs, min_scale=min_scale)

    expected = sun_scale.chan_lambda.to(u.m).value / 0.1
    assert np.allclose(sun_scale.min_scale_chan_lambda.to(u.m).value, expected)


def _build_uvws(ms_path: Path) -> UVWs:
    baselines = get_baselines_from_ms(ms_path=ms_path)
    hour_angles = make_hour_angles_for_ms(ms_path=ms_path, position="sun")
    return xyz_to_uvw(baselines=baselines, hour_angles=hour_angles)


def test_uvw_flagger_dry_run_leaves_flags(ms_example: Path) -> None:
    """A dry run walks the plank but touches no FLAGs"""
    uvws = _build_uvws(ms_example)

    with table(str(ms_example), ack=False) as tab:
        before = tab.getcol("FLAG").sum()

    result = uvw_flagger(computed_uvws=uvws, dry_run=True)

    assert result == ms_example
    with table(str(ms_example), ack=False) as tab:
        after = tab.getcol("FLAG").sum()
    assert before == after


def test_uvw_flagger_applies_flags(ms_example: Path) -> None:
    """Applying only ever adds flags (they are OR-ed into FLAG)"""
    uvws = _build_uvws(ms_example)

    with table(str(ms_example), ack=False) as tab:
        before = tab.getcol("FLAG").sum()

    result = uvw_flagger(computed_uvws=uvws, dry_run=False)

    assert result == ms_example
    with table(str(ms_example), ack=False) as tab:
        after = tab.getcol("FLAG").sum()
    assert after >= before
