"""Calculating the UVWs for a measurement set towards a direction"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.constants import c as speed_of_light
from casacore.tables import table, taql
from tqdm import tqdm

from jolly_roger.baselines import Baselines, get_baselines_from_ms
from jolly_roger.hour_angles import PositionHourAngles, make_hour_angles_for_ms
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
    time_map: dict[u.Quantity, int]
    """The mapping between time (MJDs from measurement set) to index"""
    elevation: u.Quantity
    """The elevation of the target object in time order of steps in the MS"""


def get_object_delay_for_ms(
    ms_path: Path,
    object_name: str = "sun",
) -> WDelays:
    # Generate the two sets of uvw coordinate objects
    baselines: Baselines = get_baselines_from_ms(ms_path=ms_path)
    hour_angles_phase = make_hour_angles_for_ms(
        ms_path=ms_path,
        position=None,  # gets the position from phase direction
    )
    uvws_phase: UVWs = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles_phase)

    hour_angles_object = make_hour_angles_for_ms(
        ms_path=ms_path,
        position=object_name,  # gets the position from phase direction
    )
    uvws_object: UVWs = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles_object)

    # Subtract the w-coordinates out. Since these uvws have
    # been computed towards different directions the difference
    # in w-coordinate is the delay distance
    w_diffs = uvws_object.uvws[2] - uvws_phase.uvws[2]

    delay_object = (w_diffs / speed_of_light).decompose()

    return WDelays(
        object_name=object_name,
        w_delays=delay_object,
        b_map=baselines.b_map,
        time_map=hour_angles_phase.time_map,
        elevation=hour_angles_object.elevation,
    )


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
) -> UVWs:
    """Generate the UVWs for a given set of baseline vectors towards a position
    across a series of hour angles.

    Args:
        baselines (Baselines): The set of baselines vectors to use
        hour_angles (PositionHourAngles): The hour angles and position to generate UVWs for

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

    with table(str(ms_path / "SPECTRAL_WINDOW")) as tab:
        chan_freqs = tab.getcol("CHAN_FREQ")[0] * u.Hz

    chan_lambda_m = np.squeeze((speed_of_light / chan_freqs).to(u.m))

    sun_min_scale_chan_lambda = chan_lambda_m / min_scale.to(u.rad).value

    return SunScale(
        min_scale_chan_lambda=sun_min_scale_chan_lambda,
        chan_lambda=chan_lambda_m,
        min_scale_deg=min_scale,
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
    hour_angles = computed_uvws.hour_angles
    baselines = computed_uvws.baselines
    ms_path = computed_uvws.baselines.ms_path

    sun_scale = get_sun_uv_scales(
        ms_path=ms_path,
        min_scale=min_sun_scale,
    )

    # A list of (ant1, ant2) to baseline index
    antennas_for_baselines = baselines.b_map.keys()
    logger.info(f"Will be considering {len(antennas_for_baselines)} baselines")

    elevation_curve = hour_angles.elevation

    # Used to capture the baseline and additional flags added
    summary: dict[tuple[int, int], BaselineFlagSummary] = {}

    logger.info(f"Opening {ms_path=}")
    with table(str(ms_path), ack=False, readonly=False) as ms_tab:
        for ant_1, ant_2 in tqdm(antennas_for_baselines):
            logger.debug(f"Processing {ant_1=} {ant_2=}")

            # Keeps the ruff from complaining about and unused varuable wheen
            # it is used in the table access command below
            _ = ms_tab

            # TODO: It is unclear to TJG whether the time-order needs
            # to be considered when reading in per-baseline at a time.
            # initial version operated on a row basis so an explicit map
            # to the t_idx of computed_uvws.uvws was needed (or useful?)

            # Get the UVWs and for the baseline and calculate the uv-distance
            b_idx = baselines.b_map[(ant_1, ant_2)]
            uvws_bt = computed_uvws.uvws[:, b_idx]
            uv_dist = np.sqrt((uvws_bt[0]) ** 2 + (uvws_bt[1]) ** 2).to(u.m).value

            # The max angular scale corresponds to the shortest uv-distance
            # The min angular scale corresponds to the longest uv-distance
            flag_uv_dist = (
                uv_dist[:, None]
                <= sun_scale.min_scale_chan_lambda.to(u.m).value[None, :]
            )
            flag_elevation = (min_horizon_lim < elevation_curve)[:, None] & (
                elevation_curve <= max_horizon_lim
            )[:, None]

            all_flags = flag_uv_dist & flag_elevation

            # Only need to interact with the MS if there are flags to update
            if not np.any(all_flags):
                continue

            baseline_summary = BaselineFlagSummary(
                uvw_flag_perc=np.sum(flag_uv_dist)
                / np.prod(flag_uv_dist.shape)
                * 100.0,
                elevation_flag_perc=np.sum(flag_elevation)
                / np.prod(flag_elevation.shape)
                * 100.0,
                jolly_flag_perc=np.sum(all_flags) / np.prod(all_flags.shape) * 100.0,
            )
            summary[(ant_1, ant_2)] = baseline_summary

            # Do not apply the flags mattteee
            if dry_run:
                continue

            with taql(
                "select from $ms_tab where ANTENNA1 == $ant_1 and ANTENNA2 == $ant_2",
            ) as subtab:
                flags = subtab.getcol("FLAG")[:]
                total_flags = flags | all_flags[..., None]

                subtab.putcol("FLAG", total_flags)
                subtab.flush()

    log_summaries(
        summary=summary,
        min_horizon_lim=min_horizon_lim,
        max_horizon_lim=max_horizon_lim,
        min_sun_scale=min_sun_scale,
        dry_run=dry_run,
    )

    return ms_path
