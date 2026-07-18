"""Calculating the UVWs for a measurement set towards a direction"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.constants import c as speed_of_light
from astropy.coordinates import SkyCoord
from casacore.tables import table, taql
from numpy.typing import NDArray
from tqdm import tqdm

from jolly_roger.baselines import (
    BaselineData,
    Baselines,
    OpenMSTables,
    get_baselines_from_ms,
    get_baselines_from_tables,
    get_open_ms_tables,
)
from jolly_roger.hour_angles import (
    PositionHourAngles,
    make_hour_angles_for_ms,
    make_hour_angles_from_tables,
)
from jolly_roger.logging import logger


@dataclass(frozen=True)
class WDelays:
    """Representation and mappings for the w-coordinate derived delays"""

    object_name: str
    """The name of the object that the delays are derived towards"""
    w_delays: u.Quantity
    """The w-derived delay. Shape is [baseline, time]"""
    b_map: dict[tuple[int, int], int]
    """The mapping between (ANTENNA1,ANTENNA2) to baseline index"""
    time_map: dict[float, int]
    """The mapping between time (MJDs from measurement set) to index"""
    elevation: u.Quantity
    """The elevation of the target object in time order of steps in the MS"""
    guard_region: u.Quantity | None = None
    """Define a guard region around the delay=0 based on a nominal field of view. Will be of shape {baseline, timestep}"""


def construct_guard_region(uvws: u.Quantity, radial_fov: u.Quantity) -> u.Quantity:
    """Construct the expected region around delay=0 to protect the field
    data from being contaminated by objects to be nulled. Implements:

    >>> theta * np.hypot(u, v) / speed_of_light

    where theta is the nominal field of view

    Args:
        uvws (u.Quantity): The UVW coordinates, in meters, towards the phase direction
        radial_fov (u.Quantity): The radial field of view to protect, in radians

    Returns:
        u.Quantity: The guard region for (baseline, timestep), matching the oordering of the ``uvws``
    """

    assert uvws.ndim == 3, f"Expected a rank 3 array, got {uvws.shape=}"
    assert uvws.shape[0] == 3, (
        f"Expected first axies to be (u,v,w), got something else, {uvws.shape=}"
    )

    logger.info(
        f"Constructing guard region around delay=0 using {radial_fov.to('deg')}"
    )

    # Appears as though the np.hyot is returning m2
    uvws_m = uvws.to("m").value
    fov = (radial_fov.to("rad").value * np.hypot(uvws_m[0], uvws_m[1])) * u.m
    fov = (fov / speed_of_light).decompose()

    assert fov.ndim == 2, f"Unexpected {fov.shape=}"
    assert fov.shape == uvws.shape[1:], (
        f"Mismatch in expected dimensions {fov.shape=} to {uvws.shape[1:]=}"
    )

    return fov


def get_object_delay_from_tables(
    open_ms_tables: OpenMSTables,
    phase_dir: SkyCoord,
    object_name: str | SkyCoord | Sequence[str | SkyCoord] = "sun",
    reverse_baselines: bool = False,
    flip_uvw_sign: bool = False,
    radial_fov: u.Quantity | None = None,
) -> list[WDelays]:
    """Calculate the object delay from already-open MS tables, so a caller
    that already holds ``open_ms_tables`` need not re-open the MS.

    Args:
        open_ms_tables (OpenMSTables): Already-open references to the MS tables
        phase_dir (SkyCoord): The phase direction (this couuls be more broadly considered direction 1)
        object_name (str | SkyCoord | Sequence[str  |  SkyCoord], optional): The collection of other sky positions to calculate the delays towards. Defaults to "sun".
        reverse_baselines (bool, optional): Whether the MS has antennas recorded as (ant1, ant2) or (ant2, ant1), where ant2 is always larger. Defaults to False.
        flip_uvw_sign (bool, optional): Indicates whether a sign slip needs to be introduced to the UVWs. Defaults to False.
        radial_fov (u.Quantity, optional): The radial angular size of the guard region of the main lobe that will be guarded in delay space. If None no guard boundaries are attached to the returned ``WDelays``.

    Returns:
        list[WDelays]: Description of the delay towards the nominated object. A list of objects will always be returned.
    """
    object_name = [object_name] if isinstance(object_name, str) else object_name
    assert isinstance(object_name, list | tuple), (
        f"Expected type list | tuple, got {type(object_name)=}"
    )

    baselines = get_baselines_from_tables(
        open_ms_tables=open_ms_tables,
        reverse_baselines=reverse_baselines,
    )
    hour_angles_phase = make_hour_angles_from_tables(
        open_ms_tables=open_ms_tables,
        position=phase_dir,
    )
    object_hour_angles = [
        (
            name,
            make_hour_angles_from_tables(open_ms_tables=open_ms_tables, position=name),
        )
        for name in object_name
    ]

    return get_object_delay(
        baselines=baselines,
        hour_angles_phase=hour_angles_phase,
        object_hour_angles=object_hour_angles,
        flip_uvw_sign=flip_uvw_sign,
        radial_fov=radial_fov,
    )


def get_object_delay_for_ms(
    ms_path: Path,
    phase_dir: SkyCoord,
    object_name: str | SkyCoord | Sequence[str | SkyCoord] = "sun",
    reverse_baselines: bool = False,
    flip_uvw_sign: bool = False,
    radial_fov: u.Quantity | None = None,
) -> list[WDelays]:
    """Calculate the delay between the phase-direction in the measuurement and a set of object directions.
    The delay is calculated by computing the UVWs in both directions and examining the difference in the
    w-term.

    Args:
        ms_path (Path): The measurement set and associated meta-data
        phase_dir (SkyCoord): The phase direction (this couuls be more broadly considered direction 1)
        object_name (str | SkyCoord | Sequence[str  |  SkyCoord], optional): The collection of other sky positions to calculate the delays towards. Defaults to "sun".
        reverse_baselines (bool, optional): Whether the MS has antennas recorded as (ant1, ant2) or (ant2, ant1), where ant2 is always larger. Defaults to False.
        flip_uvw_sign (bool, optional): Indicates whether a sign slip needs to be introduced to the UVWs. Defaults to False.
        radial_fov (u.Quantity, optional): The radial angular size of the guard region of the main lobe that will be guarded in delay space. If None no guard boundaries are attached to the returned ``WDelays``.

    Returns:
        list[WDelays]: Description of the delay towards the nominated object. A list of objects will always be returned.
    """
    with get_open_ms_tables(ms_path) as open_ms_tables:
        return get_object_delay_from_tables(
            open_ms_tables=open_ms_tables,
            phase_dir=phase_dir,
            object_name=object_name,
            reverse_baselines=reverse_baselines,
            flip_uvw_sign=flip_uvw_sign,
            radial_fov=radial_fov,
        )


def get_object_delay(
    baselines: Baselines,
    hour_angles_phase: PositionHourAngles,
    object_hour_angles: Sequence[tuple[str | SkyCoord, PositionHourAngles]],
    flip_uvw_sign: bool = False,
    radial_fov: u.Quantity | None = None,
) -> list[WDelays]:
    """Compute the w-derived delays between the phase direction and each object.

    Args:
        baselines (Baselines): The baseline vectors
        hour_angles_phase (PositionHourAngles): Hour angles towards the phase direction
        object_hour_angles (Sequence[tuple[str | SkyCoord, PositionHourAngles]]): The (name, hour angles) for each object
        flip_uvw_sign (bool, optional): Flip the sign of the UVWs. Defaults to False.
        radial_fov (u.Quantity | None, optional): Radial FoV for the guard region. Defaults to None.

    Returns:
        list[WDelays]: The delay towards each nominated object
    """
    # Generate the two sets of uvw coordinate objects
    uvws_phase = xyz_to_uvw(
        baselines=baselines, hour_angles=hour_angles_phase, flip_uvw_sign=flip_uvw_sign
    )

    guard_region: None | u.Quantity = None
    if radial_fov is not None:
        guard_region = construct_guard_region(
            uvws=uvws_phase.uvws, radial_fov=radial_fov
        )

    object_w_delays: list[WDelays] = []

    for object_name, hour_angles_object in object_hour_angles:
        uvws_object = xyz_to_uvw(
            baselines=baselines,
            hour_angles=hour_angles_object,
            flip_uvw_sign=flip_uvw_sign,
        )

        # Subtract the w-coordinates out. Since these uvws have
        # been computed towards different directions the difference
        # in w-coordinate is the delay distance
        w_diffs = uvws_object.uvws[2] - uvws_phase.uvws[2]

        delay_for_object = (w_diffs / speed_of_light).decompose()

        w_delay = WDelays(
            object_name=object_name,
            w_delays=delay_for_object,
            b_map=baselines.b_map,
            time_map=hour_angles_phase.time_map,
            elevation=hour_angles_object.elevation,
            guard_region=guard_region,
        )
        logger.info(f"Have created for {w_delay.object_name}")
        object_w_delays.append(w_delay)

    return object_w_delays


@dataclass(frozen=True)
class WDelayRates:
    """Representation and mappings for the w-coordinate derived delays"""

    object_name: str
    """The name of the object that the delays are derived towards"""
    w_delays: u.Quantity
    """The w-derived delay. Shape is [baseline, time]"""
    w_rates: u.Quantity
    """The w-derived rates. Shape is [baseline, time]"""
    b_map: dict[tuple[int, int], int]
    """The mapping between (ANTENNA1,ANTENNA2) to baseline index"""
    time_map: dict[float, int]
    """The mapping between time (MJDs from measurement set) to index"""
    elevation: u.Quantity
    """The elevation of the target object in time order of steps in the MS"""


def get_object_delayrate_for_baseline(
    baseline_data: BaselineData,
    object_name: str | list[str] = "sun",
    reverse_baselines: bool = False,
) -> list[WDelayRates]:
    object_name = [object_name] if isinstance(object_name, str) else object_name
    assert isinstance(object_name, list), (
        f"Expected type list, got {type(object_name)=}"
    )
    assert baseline_data.ms_path is not None, "baseline_data has no ms_path"
    ms_path = baseline_data.ms_path

    # Generate the two sets of uvw coordinate objects
    baselines = get_baselines_from_ms(
        ms_path=ms_path,
        reverse_baselines=reverse_baselines,
    )
    hour_angles_phase = make_hour_angles_for_ms(
        ms_path=ms_path,
        position=None,  # gets the position from phase direction
    )
    uvws_phase = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles_phase)

    baseline_idx = baselines.b_map[(baseline_data.ant_1, baseline_data.ant_2)]

    object_w_delay_rates: list[WDelayRates] = []

    for _object_name in object_name:
        hour_angles_object = make_hour_angles_for_ms(
            ms_path=ms_path,
            position=_object_name,  # gets the position from phase direction
        )
        uvws_object = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles_object)

        # Subtract the w-coordinates out. Since these uvws have
        # been computed towards different directions the difference
        # in w-coordinate is the delay distance
        w_diffs = (uvws_object.uvws[2] - uvws_phase.uvws[2])[baseline_idx]

        delay_for_object = (w_diffs / speed_of_light).decompose()

        mean_freq = baseline_data.freq_chan.mean().to(u.Hz)

        delay_rate_for_object = (
            np.gradient(delay_for_object, (hour_angles_object.time.jd * u.day).to(u.s))
            * mean_freq
        )  # s / s * Hz

        w_delay_rate = WDelayRates(
            object_name=_object_name,
            w_delays=delay_for_object,
            w_rates=delay_rate_for_object,
            b_map=baselines.b_map,
            time_map=hour_angles_phase.time_map,
            elevation=hour_angles_object.elevation,
        )
        logger.info(f"Have created for {w_delay_rate.object_name}")
        object_w_delay_rates.append(w_delay_rate)

    return object_w_delay_rates


@dataclass
class UVWs:
    """A small container to represent uvws"""

    uvws: np.ndarray
    """The (U,V,W) coordinatesm shape [coord, baseline, time]"""
    hour_angles: PositionHourAngles
    """The hour angle information used to construct the UVWs"""
    baselines: Baselines
    """The set of antenna baselines used for form the UVWs"""


def xyz_to_uvw(
    baselines: Baselines,
    hour_angles: PositionHourAngles,
    flip_uvw_sign: bool = False,
) -> UVWs:
    """Generate the UVWs for a given set of baseline vectors towards a position
    across a series of hour angles.

    Args:
        baselines (Baselines): The set of baselines vectors to use
        hour_angles (PositionHourAngles): The hour angles and position to generate UVWs for
        flip_uvw_sign (bool, optional): Flip the UVWs (required for LOFAR). Defaults to False.

    Returns:
        UVWs: The generated set of UVWs
    """
    b_xyz = baselines.b_xyz

    # Convert HA to geocentric hour angle (at Greenwich meridian)
    # This is why we subtract the location's longitude
    ha = hour_angles.hour_angle - hour_angles.location.lon

    declination = hour_angles.position.dec

    # This is necessary for broadcastung in the matrix to work.
    # Should the position be a solar object like the sun its position
    # will change throughout the observation. but it will have
    # been created consistently with the hour angles. If it is fixed
    # then the use of the numpy ones like will ensure the same shape.
    declination = (np.ones(len(ha)) * declination).decompose()

    # Precompute the repeated terms
    sin_ha = np.sin(ha)
    sin_dec = np.sin(declination)
    cos_ha = np.cos(ha)
    cos_dec = np.cos(declination)
    zeros = np.zeros_like(sin_ha)

    # Conversion from baseline vectors to UVW
    mat = np.array(
        [
            [sin_ha, cos_ha, zeros],
            [-sin_dec * cos_ha, sin_dec * sin_ha, cos_dec],
            [
                cos_dec * cos_ha,
                -cos_dec * sin_ha,
                sin_dec,
            ],
        ]
    )

    # Every time this confuses me and I need the first mate to look over.
    # b_xyz shape: (baselines, 3) where coord is XYZ
    # mat shape: (3, 3, timesteps)
    # uvw shape: (3, baseline, timesteps) where coord is UVW
    uvw = np.einsum("ijk,lj->ilk", mat, b_xyz, optimize=True)  # codespell:ignore ilk
    # i,j,k -> (3, 3, time)
    # l,j -> (baseline, 3)
    # i,l,k -> (3, baseline, time)

    logger.debug(f"{uvw.shape=}")

    if flip_uvw_sign:
        logger.warning("Flipping sign of UVWs!")
        uvw *= -1

    return UVWs(uvws=uvw, hour_angles=hour_angles, baselines=baselines)


@dataclass
class SunScale:
    """Describes the (u,v)-scales sensitive to angular scales of the Sun"""

    min_scale_chan_lambda: u.Quantity
    """The distance that corresponds to an angular scale scaled to each channel set using the minimum angular scale"""
    chan_lambda: u.Quantity
    """The wavelength of each channel"""
    min_scale_deg: float
    """The minimum angular scale used for baseline flagging"""


def compute_sun_uv_scales(
    chan_freqs: u.Quantity,
    min_scale: u.Quantity = 0.075 * u.deg,
) -> SunScale:
    """Angular scales and the corresponding (u,v)-distances sensitive to them.

    Args:
        chan_freqs (u.Quantity): The per-channel frequencies
        min_scale (u.Quantity, optional): The minimum angular scale to project. Defaults to 0.075*u.deg.

    Returns:
        SunScale: The sun scales in distances
    """
    chan_lambda_m = np.squeeze((speed_of_light / chan_freqs).to(u.m))

    sun_min_scale_chan_lambda = chan_lambda_m / min_scale.to(u.rad).value

    return SunScale(
        min_scale_chan_lambda=sun_min_scale_chan_lambda,
        chan_lambda=chan_lambda_m,
        min_scale_deg=min_scale,
    )


def get_sun_uv_scales_from_tables(
    open_ms_tables: OpenMSTables,
    min_scale: u.Quantity = 0.075 * u.deg,
) -> SunScale:
    """Sun (u,v)-scales from the CHAN_FREQ of open MS tables."""
    chan_freqs = open_ms_tables.spw_table.getcol("CHAN_FREQ")[0] * u.Hz
    return compute_sun_uv_scales(chan_freqs=chan_freqs, min_scale=min_scale)


def get_sun_uv_scales(
    ms_path: Path,
    min_scale: u.Quantity = 0.075 * u.deg,
) -> SunScale:
    """Compute the angular scales and the corresponding (u,v)-distances that
    would be sensitive to them.

    Args:
        ms_path (Path): The measurement set to consider, where frequency information is extracted from
        min_scale (u.Quantity, optional): The minimum angular scale that will be projected and flgged. Defaults to 0.075*u.deg.

    Returns:
        SunScale: The sun scales in distances
    """
    with get_open_ms_tables(ms_path) as open_ms_tables:
        return get_sun_uv_scales_from_tables(
            open_ms_tables=open_ms_tables, min_scale=min_scale
        )


@dataclass
class BaselineFlagSummary:
    """Container to capture the flagged baselines statistics"""

    uvw_flag_perc: float
    """The percentage of flags to add based on the uv-distance cut"""
    elevation_flag_perc: float
    """The percentage of flags to add based on the elevation cut"""
    jolly_flag_perc: float
    """The percentage of new to add based on both criteria"""


def log_summaries(
    summary: dict[tuple[int, int], BaselineFlagSummary],
    min_horizon_lim: u.Quantity,
    max_horizon_lim: u.Quantity,
    min_sun_scale: u.Quantity,
    dry_run: bool = False,
) -> None:
    """Log the flagging statistics made throughout the `uvw_flagger`.

    Args:
        summary (dict[tuple[int, int], BaselineFlagSummary]): Collection of flagging statistics accumulated when flagging
        min_horizon_lim (u.Quantity): The minimum horizon limit applied to the flagging.
        max_horizon_lim (u.Quantity): The maximum horizon limit applied to the flagging.
        min_sun_scale (u.Quantity): The sun scale used to compute the uv-distance limiter.
        dry_run (bool, optional): Indicates whether the flags were applied. Defaults to False.

    """
    logger.info("----------------------------------")
    logger.info("Flagging summary of modified flags")
    logger.info(f"Minimum Horizon Limit: {min_horizon_lim}")
    logger.info(f"Maximum Horizon Limit: {max_horizon_lim}")
    logger.info(f"Minimum Sun Scale: {min_sun_scale}")
    if dry_run:
        logger.info("(Dry run, not applying)")
    logger.info("----------------------------------")

    for ants, baseline_summary in summary.items():
        logger.info(
            f"({ants[0]:3d},{ants[1]:3d}): uvw {baseline_summary.uvw_flag_perc:>6.2f}% & elev. {baseline_summary.elevation_flag_perc:>6.2f}% = Applied {baseline_summary.jolly_flag_perc:>6.2f}%"
        )

    logger.info("\n")


@dataclass
class UVWFlagResult:
    """The per-baseline flag decision from the (u,v,w) and sun scales"""

    flags: dict[tuple[int, int], NDArray[np.bool_]]
    """Per-baseline (time, chan) flag mask, only for baselines with flags to add"""
    summary: dict[tuple[int, int], BaselineFlagSummary]
    """Per-baseline flagging statistics, keyed as ``flags``"""


def compute_uvw_flags(
    computed_uvws: UVWs,
    sun_scale: SunScale,
    min_horizon_lim: u.Quantity = -3 * u.deg,
    max_horizon_lim: u.Quantity = 90 * u.deg,
) -> UVWFlagResult:
    """Decide which visibilities to flag from the projected baselines and sun scales.

    The baseline length affected by the Sun is compared against the projected
    (u,v)-distance; visibilities are flagged where they are presumably sensitive
    to the Sun and the Sun is within the horizon limits.

    Args:
        computed_uvws (UVWs): The pre-computed UVWs and associated meta-data
        sun_scale (SunScale): The sun (u,v)-scales to compare against
        min_horizon_lim (u.Quantity, optional): The lower horizon limit. Defaults to -3*u.deg.
        max_horizon_lim (u.Quantity, optional): The upper horizon limit. Defaults to 90*u.deg.

    Returns:
        UVWFlagResult: The per-baseline flag masks and statistics
    """
    baselines = computed_uvws.baselines
    elevation_curve = computed_uvws.hour_angles.elevation

    flags: dict[tuple[int, int], NDArray[np.bool_]] = {}
    summary: dict[tuple[int, int], BaselineFlagSummary] = {}

    for ant_1, ant_2 in baselines.b_map:
        logger.debug(f"Processing {ant_1=} {ant_2=}")

        # Get the UVWs and for the baseline and calculate the uv-distance
        b_idx = baselines.b_map[(ant_1, ant_2)]
        uvws_bt = computed_uvws.uvws[:, b_idx]
        uv_dist = np.sqrt((uvws_bt[0]) ** 2 + (uvws_bt[1]) ** 2).to(u.m).value

        # The max angular scale corresponds to the shortest uv-distance
        # The min angular scale corresponds to the longest uv-distance
        flag_uv_dist = (
            uv_dist[:, None] <= sun_scale.min_scale_chan_lambda.to(u.m).value[None, :]
        )
        flag_elevation = (min_horizon_lim < elevation_curve)[:, None] & (
            elevation_curve <= max_horizon_lim
        )[:, None]

        all_flags = flag_uv_dist & flag_elevation

        # Only record baselines that actually gain flags
        if not np.any(all_flags):
            continue

        summary[(ant_1, ant_2)] = BaselineFlagSummary(
            uvw_flag_perc=np.sum(flag_uv_dist) / np.prod(flag_uv_dist.shape) * 100.0,
            elevation_flag_perc=np.sum(flag_elevation)
            / np.prod(flag_elevation.shape)
            * 100.0,
            jolly_flag_perc=np.sum(all_flags) / np.prod(all_flags.shape) * 100.0,
        )
        flags[(ant_1, ant_2)] = all_flags

    return UVWFlagResult(flags=flags, summary=summary)


def uvw_flagger(
    computed_uvws: UVWs,
    min_horizon_lim: u.Quantity = -3 * u.deg,
    max_horizon_lim: u.Quantity = 90 * u.deg,
    min_sun_scale: u.Quantity = 0.075 * u.deg,
    dry_run: bool = False,
) -> Path:
    """Flag visibilities based on the (u, v, w)'s and assumed scales of
    the sun. The routine will compute ht ebaseline length affected by the Sun
    and then flagged visibilities where the projected (u,v)-distance towards
    the direction of the Sun and presumably sensitive.

    Args:
        computed_uvws (UVWs): The pre-computed UVWs and associated meta-data
        min_horizon_lim (u.Quantity, optional): The lower horixzon limit required for flagging to be applied. Defaults to -3*u.deg.
        max_horizon_lim (u.Quantity, optional): The upper horixzon limit required for flagging to be applied. Defaults to 90*u.deg.
        min_sun_scale (u.Quantity, options): The minimum angular scale to consider when flagging the projected baselines. Defaults to 0.075*u.deg.
        dry_run (bool, optional): Do not apply the flags to the measurement set. Defaults to False.


    Returns:
        Path: The path to the flagged measurement set
    """
    baselines = computed_uvws.baselines
    assert baselines.ms_path is not None, "baselines has no ms_path"
    ms_path = baselines.ms_path

    sun_scale = get_sun_uv_scales(ms_path=ms_path, min_scale=min_sun_scale)

    logger.info(f"Will be considering {len(baselines.b_map)} baselines")

    result = compute_uvw_flags(
        computed_uvws=computed_uvws,
        sun_scale=sun_scale,
        min_horizon_lim=min_horizon_lim,
        max_horizon_lim=max_horizon_lim,
    )

    log_summaries(
        summary=result.summary,
        min_horizon_lim=min_horizon_lim,
        max_horizon_lim=max_horizon_lim,
        min_sun_scale=min_sun_scale,
        dry_run=dry_run,
    )

    # Do not apply the flags mattteee
    if dry_run:
        return ms_path

    logger.info(f"Opening {ms_path=}")
    with table(str(ms_path), ack=False, readonly=False) as ms_tab:
        for (ant_1, ant_2), all_flags in tqdm(result.flags.items()):
            # Keeps ruff from complaining about variables that are only
            # referenced inside the taql string below
            _ = ms_tab, ant_1, ant_2

            with taql(
                "select from $ms_tab where ANTENNA1 == $ant_1 and ANTENNA2 == $ant_2",
            ) as subtab:
                flags = subtab.getcol("FLAG")[:]
                total_flags = flags | all_flags[..., None]

                subtab.putcol("FLAG", total_flags)
                subtab.flush()

    return ms_path
