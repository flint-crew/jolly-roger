"""Tests around the UVW delays, sun scales and flagging"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
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
    WDelays,
    compute_sun_uv_scales,
    compute_uvw_flags,
    construct_rate_guard_region,
    get_object_delay_for_ms,
    get_w_rates,
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


def test_construct_rate_guard_region() -> None:
    """For a uv-track moving at a constant speed the guard is theta * speed / c"""
    n_baseline, n_time = 2, 10
    time_s = np.arange(n_time) * 10.0
    speed_m_s = np.array([1.0, 3.0])
    uvws = np.zeros((3, n_baseline, n_time))
    uvws[0] = speed_m_s[:, None] * time_s[None, :] * 0.6
    uvws[1] = speed_m_s[:, None] * time_s[None, :] * 0.8

    radial_fov = 1.0 * u.deg
    guard = construct_rate_guard_region(
        uvws=uvws * u.m, time_s=time_s, radial_fov=radial_fov
    )

    assert guard.shape == (n_baseline, n_time)
    expected = np.deg2rad(1.0) * speed_m_s / 299792458.0
    np.testing.assert_allclose(guard.value, np.repeat(expected[:, None], n_time, 1))

    double = construct_rate_guard_region(
        uvws=uvws * u.m, time_s=time_s, radial_fov=2 * radial_fov
    )
    np.testing.assert_allclose(double.value, 2 * guard.value)


def test_get_w_rates_derived_from_delays() -> None:
    """Without attached rates they are derived from the delays and time map"""
    time_s = 5e9 + np.arange(5) * 10.0
    w_delays = WDelays(
        object_name="sun",
        w_delays=(2e-11 * (time_s - time_s[0]) * u.s)[None, :],
        b_map={(0, 1): 0},
        time_map={t * u.s: idx for idx, t in enumerate(time_s)},
        elevation=np.full(5, 45.0) * u.deg,
    )
    np.testing.assert_allclose(get_w_rates(w_delays).value, 2e-11)

    attached = WDelays(
        object_name="sun",
        w_delays=w_delays.w_delays,
        b_map=w_delays.b_map,
        time_map=w_delays.time_map,
        elevation=w_delays.elevation,
        w_rates=np.full((1, 5), 7.0) * u.dimensionless_unscaled,
    )
    np.testing.assert_allclose(get_w_rates(attached).value, 7.0)


def test_get_object_delay_attaches_rates(ms_example: Path) -> None:
    with table(str(ms_example / "FIELD"), ack=False) as tab:
        phase_dir = tab.getcol("PHASE_DIR")[0, 0]
    phase = SkyCoord(*phase_dir, unit="rad")

    (without_guard,) = get_object_delay_for_ms(
        ms_path=ms_example, phase_dir=phase, object_name="sun"
    )
    assert without_guard.w_rates is not None
    assert without_guard.w_rates.shape == without_guard.w_delays.shape
    assert without_guard.rate_guard_region is None
    # Rates should be consistent with the delays
    np.testing.assert_allclose(
        without_guard.w_rates.value,
        get_w_rates(
            WDelays(
                object_name="sun",
                w_delays=without_guard.w_delays,
                b_map=without_guard.b_map,
                time_map=without_guard.time_map,
                elevation=without_guard.elevation,
            )
        ).value,
    )

    (with_guard,) = get_object_delay_for_ms(
        ms_path=ms_example, phase_dir=phase, object_name="sun", radial_fov=1 * u.deg
    )
    assert with_guard.rate_guard_region is not None
    assert with_guard.rate_guard_region.shape == with_guard.w_delays.shape
    assert np.all(with_guard.rate_guard_region.value >= 0)


def test_get_indices_matches_dict_lookup(ms_example: Path) -> None:
    """The vectorised lookup agrees with per-row lookups of the maps"""
    with table(str(ms_example / "FIELD"), ack=False) as tab:
        phase_dir = tab.getcol("PHASE_DIR")[0, 0]
    (w_delays,) = get_object_delay_for_ms(
        ms_path=ms_example, phase_dir=SkyCoord(*phase_dir, unit="rad")
    )
    with table(str(ms_example), ack=False) as tab:
        ant_1 = tab.getcol("ANTENNA1")
        ant_2 = tab.getcol("ANTENNA2")
        time_mjds = tab.getcol("TIME_CENTROID")

    baseline_idx, time_idx = w_delays.get_indices(
        ant_1=ant_1, ant_2=ant_2, time_mjds=time_mjds
    )

    expected_baseline_idx = [
        w_delays.b_map[(int(a1), int(a2))] if a1 != a2 else 0
        for a1, a2 in zip(ant_1, ant_2, strict=True)
    ]
    expected_time_idx = [w_delays.time_map[t * u.s] for t in time_mjds]
    np.testing.assert_array_equal(baseline_idx, expected_baseline_idx)
    np.testing.assert_array_equal(time_idx, expected_time_idx)


def _indexed_w_delays() -> WDelays:
    # Times deliberately out of order, as the time map is in 'first seen' order
    times = np.array([30.0, 10.0, 20.0])
    return WDelays(
        object_name="sun",
        w_delays=np.zeros((2, 3)) * u.s,
        b_map={(0, 1): 0, (0, 2): 1},
        time_map={t * u.s: idx for idx, t in enumerate(times)},
        elevation=np.zeros(3) * u.deg,
    )


def test_get_indices() -> None:
    w_delays = _indexed_w_delays()
    baseline_idx, time_idx = w_delays.get_indices(
        ant_1=np.array([0, 0, 1]),
        ant_2=np.array([2, 1, 1]),
        time_mjds=np.array([10.0, 30.0, 20.0]),
    )
    np.testing.assert_array_equal(baseline_idx, [1, 0, 0])
    np.testing.assert_array_equal(time_idx, [1, 0, 2])


@pytest.mark.parametrize(
    ("ant_1", "ant_2", "time_mjds"),
    [
        ([1], [2], [10.0]),  # unknown baseline
        ([0], [5], [10.0]),  # antenna beyond the map
        ([0], [1], [15.0]),  # unknown time
        ([0], [1], [40.0]),  # time beyond the map
    ],
)
def test_get_indices_unknown(ant_1, ant_2, time_mjds) -> None:
    with pytest.raises(KeyError):
        _indexed_w_delays().get_indices(
            ant_1=np.array(ant_1), ant_2=np.array(ant_2), time_mjds=np.array(time_mjds)
        )


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
