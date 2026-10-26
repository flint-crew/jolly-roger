"""Tests around the UVW delays, sun scales and flagging"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from casacore.tables import table

from jolly_roger.baselines import Baselines, get_baselines, get_baselines_from_ms
from jolly_roger.hour_angles import (
    PositionHourAngles,
    get_location,
    make_hour_angles,
    make_hour_angles_for_ms,
)
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


def _load_sb56289_uvw() -> dict[str, np.ndarray]:
    """Rows of ASKAP SB56289 (RACS_1041+18, beam 10) with the UVWs written by the
    correlator, for three timesteps"""
    resource = files("jolly_roger.data").joinpath("tests", "sb56289_uvw.npz")
    with as_file(resource) as path, np.load(path) as arr:
        return dict(arr)


def test_xyz_to_uvw_matches_askap_uvw() -> None:
    """The computed UVWs reproduce those of an ASKAP measurement set, which are
    geometric J2000 UVWs evaluated at TIME_CENTROID"""
    ref = _load_sb56289_uvw()
    ant_xyz = ref["ant_xyz"]
    phase_dir = SkyCoord(*ref["phase_dir"], unit="rad")

    baselines = get_baselines(ant_xyz)
    hour_angles = make_hour_angles(
        times_mjds=ref["time_centroid"] * u.s,
        location=get_location(ant_xyz),
        position=phase_dir,
    )
    uvws = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles).uvws.to(u.m).value

    baseline_idx = [
        baselines.b_map[(int(a1), int(a2))]
        for a1, a2 in zip(ref["ant_1"], ref["ant_2"], strict=True)
    ]
    time_idx = [hour_angles.time_map[t * u.s] for t in ref["time_centroid"]]
    computed = uvws[:, baseline_idx, time_idx].T

    # Baselines are up to 6.4 km
    np.testing.assert_allclose(computed, ref["uvw"], rtol=0, atol=1e-3)


def test_xyz_to_uvw_moving_position() -> None:
    """A position that changes with time (the Sun, in the GCRS) is projected
    with its own direction at each time"""
    ref = _load_sb56289_uvw()
    ant_xyz = ref["ant_xyz"]
    baselines = get_baselines(ant_xyz)
    times_mjds = np.unique(ref["time_centroid"]) * u.s
    location = get_location(ant_xyz)

    moving = xyz_to_uvw(
        baselines=baselines,
        hour_angles=make_hour_angles(
            times_mjds=times_mjds, location=location, position="sun"
        ),
    ).uvws
    for idx, time_mjd in enumerate(times_mjds):
        fixed_sun = get_sun(Time(time_mjd.to(u.day), format="mjd", scale="utc"))
        single = xyz_to_uvw(
            baselines=baselines,
            hour_angles=make_hour_angles(
                times_mjds=time_mjd[None], location=location, position=fixed_sun
            ),
        ).uvws
        np.testing.assert_allclose(moving[:, :, idx], single[:, :, 0])


def test_xyz_to_uvw_frame_independent() -> None:
    """The same direction given in different frames yields the same UVWs"""
    ref = _load_sb56289_uvw()
    ant_xyz = ref["ant_xyz"]
    baselines = get_baselines(ant_xyz)
    times_mjds = np.unique(ref["time_centroid"]) * u.s
    location = get_location(ant_xyz)
    icrs = SkyCoord(*ref["phase_dir"], unit="rad")

    def _uvws(position: SkyCoord) -> np.ndarray:
        return (
            xyz_to_uvw(
                baselines=baselines,
                hour_angles=make_hour_angles(
                    times_mjds=times_mjds, location=location, position=position
                ),
            )
            .uvws.to(u.m)
            .value
        )

    # FK5 (J2000) and the ICRS differ by ~20 mas
    np.testing.assert_allclose(_uvws(icrs.fk5), _uvws(icrs), atol=1e-3)
    np.testing.assert_allclose(
        xyz_to_uvw(
            baselines=baselines,
            hour_angles=make_hour_angles(
                times_mjds=times_mjds, location=location, position=icrs
            ),
            flip_uvw_sign=True,
        )
        .uvws.to(u.m)
        .value,
        -_uvws(icrs),
    )
