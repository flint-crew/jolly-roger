"""Calculating the UVWs for a measurement set towards a direction"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.constants import c as speed_of_light
from casacore.tables import table, taql

from jolly_roger.baselines import Baselines
from jolly_roger.hour_angles import PositionHourAngles
from jolly_roger.logging import logger


@dataclass
class UVWs:
    """A small container to represent uvws"""

    uvws: np.ndarray
    """The (U,V,W) coordinates"""
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

    # Getting the units right is important, mate
    ha = hour_angles.hour_angle
    ha = ha.to(u.rad)

    declination = hour_angles.position.dec
    declination = declination.to(u.rad)

    # This is necessary for broadcastung in the matrix to work.
    # Should the position be a solar object like the sub its position
    # will change throughout the observation. but it will have
    ## been created consistently with the hour angles. If it is fixed
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
                np.cos(declination) * np.cos(ha),
                np.cos(declination) * np.sin(ha),
                np.sin(declination),
            ],
        ]
    )

    # Every time this confuses me and I need the first mate to look over.
    # b_xyz shape: (baselines, coord) where coord is XYZ
    # mat shape: (3, 3, timesteps)
    # uvw shape: (baseline, coord, timesteps) where coord is UVW
    uvw = np.einsum("ijk,lj->lik", mat, b_xyz, optimize=True)  # codespell:ignore lik

    # Make order (coord, baseline, timesteps)
    uvw = np.swapaxes(uvw, 0, 1)
    logger.debug(f"{uvw.shape=}")

    return UVWs(uvws=uvw, hour_angles=hour_angles, baselines=baselines)


@dataclass
class SunScale:
    """Describes the (u,v)-scales sensitive to angular scales of the Sun"""

    sun_scale_chan_lambda: u.Quantity
    """The distance that corresponds to an angular scale scaled to each channel"""
    chan_lambda: u.Quantity
    """The wavelength of each channel"""
    minimum_scale_deg: float
    """The minimum angular scale used for baseline flagging"""


def get_sun_uv_scales(ms_path: Path, minimum_scale_deg: float = 0.075) -> SunScale:
    """Compute the angular scales and the corresponding (u,v)-distances that
    would be sensitive to them.

    Args:
        ms_path (Path): The measurement set to consider, where frequency information is extracted from
        minimum_scale_deg (float, optional): The minimum angular scale that will be projected and flgged. Defaults to 0.075.

    Returns:
        SunScale: The sun scales in distances
    """

    with table(str(ms_path / "SPECTRAL_WINDOW")) as tab:
        chan_freqs = tab.getcol("CHAN_FREQ")[0] * u.Hz

    chan_lambda_m = np.squeeze((speed_of_light / chan_freqs).to(u.m))

    sun_diameter = minimum_scale_deg * u.deg
    sun_scale_chan_lambda = chan_lambda_m / sun_diameter.to(u.rad).value

    return SunScale(
        sun_scale_chan_lambda=sun_scale_chan_lambda,
        chan_lambda=chan_lambda_m,
        minimum_scale_deg=minimum_scale_deg,
    )


def uvw_flagger(computed_uvws: UVWs, horizon_lim: u.Quantity = -3 * u.deg) -> Path:
    """Flag visibilities based on the (u, v, w)'s and assumed scales of
    the sun. The routine will compute ht ebaseline length affected by the Sun
    and then flagged visibilities where the projected (u,v)-distance towards
    the direction of the Sun and presumably sensitive.

    Args:
        computed_uvws (UVWs): The pre-computed UVWs and associated meta-data
        horizon_lim (u.Quantity, optional): The horixzon limit required for flagging to be applied. Defaults to -3*u.deg.

    Returns:
        Path: The path to the flagged measurement set
    """
    hour_angles = computed_uvws.hour_angles
    baselines = computed_uvws.baselines
    ms_path = computed_uvws.baselines.ms_path

    sun_scale = get_sun_uv_scales(ms_path=ms_path)

    # A list of (ant1, ant2) to baseline index
    antennas_for_baselines = baselines.b_map.keys()
    logger.info(f"Will be considering {len(antennas_for_baselines)} baselines")

    logger.info(f"Opening {ms_path=}")
    with table(str(ms_path), ack=False, readonly=False) as ms_tab:
        for ant_1, ant_2 in antennas_for_baselines:
            logger.debug(f"Processing {ant_1=} {ant_2=}")

            # Keeps the ruff from complaining about and unused varuable wheen
            # it is used in the table access command below
            _ = ms_tab
            with taql(
                "select from $ms_tab where ANTENNA1 == $ant_1 and ANTENNA2 == $ant_2",
            ) as subtab:
                time = subtab.getcol("TIME_CENTROID")[:]
                flags = subtab.getcol("FLAG")[:]
                logger.debug(f"{time.shape=}")
                logger.debug(f"{flags.shape=}")

                # Get the UVWs and for the baseline and calculate the uv-distance
                b_idx = baselines.b_map[(ant_1, ant_2)]
                uvws_bt = computed_uvws.uvws[:, b_idx]
                uv_dist = np.sqrt((uvws_bt[0]) ** 2 + (uvws_bt[1]) ** 2).to(u.m).value

                flag_uv_dist = (
                    uv_dist[:, None]
                    < sun_scale.sun_scale_chan_lambda.to(u.m).value[None, :]
                ) & (hour_angles.elevation > horizon_lim)[:, None]

                total_flags = np.logical_or(flags, flag_uv_dist[..., None])

                subtab.putcol("FLAG", total_flags)
                subtab.flush()

    return ms_path
