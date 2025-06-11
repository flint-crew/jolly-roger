"""Utilities to construct properties that scale over hour angle"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from casacore.tables import table

from jolly_roger.logging import logger

# Default location with XYZ based on mean of antenna positions
ASKAP_XYZ_m = np.array([-2556146.66356375, 5097426.58592797, -2848333.08164107]) * u.m
ASKAP = EarthLocation(*ASKAP_XYZ_m)


@dataclass
class PositionHourAngles:
    """Represent time, hour angles and other quantities for some
    assumed sky position. Time intervals are intended to represent
    those stored in a measurement set."""

    hour_angle: u.rad
    """The hour angle across sampled time intervales of a source for a Earth location"""
    time_mjds: np.ndarray
    """The MJD time in seconds from which other quantities are evalauted against. Should be drawn from a measurement set."""
    location: EarthLocation
    """The location these quantities have been derived from."""
    position: SkyCoord
    """The sky-position that is being used to calculate quantities towards"""
    elevation: np.ndarray
    """The elevation of the ``position` direction across time"""
    time: Time
    """Representation of the `time_mjds` attribute"""
    time_map: dict[float, int]
    """Index mapping of time steps described in `time_mjds` to an array index position.
    This is done by selecting all unique `time_mjds` and ordering in this 'first seen'
    position"""


def _process_position(
    position: SkyCoord | Literal["sun"] | None = None,
    ms_path: Path | None = None,
    times: Time | None = None,
) -> SkyCoord:
    """Acquire a SkyCoord object towards a specified position. If
    a known string position is provided this will be looked up and
    may required the `times` (e.g. for the sun). Otherwise is position
    is None it will be drawn from the PHASE_DIR in the provided measurement
    set

    Args:
        position (SkyCoord | Literal[&quot;sun&quot;] | None, optional): The position to be considered. Defaults to None.
        ms_path (Path | None, optional): The path with the PHASE_DIR to use should `position` be None. Defaults to None.
        times (Time | None, optional): Times to used if they are required in the lookup. Defaults to None.

    Raises:
        ValueError: Raised if a string position is provided without a `times`
        ValueError: Raised is position is None and no ms_path provided
        ValueError: Raised if no final SkyCoord is constructed

    Returns:
        SkyCoord: The position to use
    """

    if isinstance(position, str):
        if times is None:
            msg = f"{times=}, but needs to be set when position is a name"
            raise ValueError(msg)
        if position == "sun":
            logger.info("Getting sky-position of the sun")
            position = get_sun(times)

    if position is None:
        if ms_path is None:
            msg = f"{position=}, so default position can't be drawn. Provide a ms_path="
            raise ValueError(msg)

        with table(str(ms_path / "FIELD")) as tab:
            logger.info(f"Getting the sky-position from PHASE_DIR of {ms_path=}")
            field_positions = tab.getcol("PHASE_DIR")
            position = SkyCoord(field_positions[0] * u.rad)

    if isinstance(position, SkyCoord):
        return position

    # Someone sea dog is having a laugh
    msg = "Something went wrong in the processing of position"
    raise ValueError(msg)


def make_hour_angles_for_ms(
    ms_path: Path,
    location: EarthLocation = ASKAP,
    position: SkyCoord | str | None = None,
    whole_day: bool = False,
) -> PositionHourAngles:
    """Calculate hour-angle and time quantities for a given position using time information
    encoded in a nominated measurement set at a nominated location

    Args:
        ms_path (Path): Measurement set to usefor time and sky-position information
        location (EarthLocation, optional): The location to use when calculate LST. Defaults to ASKAP.
        position (SkyCoord | str | None, optional): The sky-direction hour-angles will be calculated towards. Defaults to None.
        whole_day (bool, optional): Calaculate for a 24 hour persion starting from the first time step. Defaults to False.

    Returns:
        PositionHourAngle: Compute hour angles, normalised times and elevation
    """

    logger.info(f"Computing hour angles for {ms_path=}")
    with table(str(ms_path), ack=False) as tab:
        logger.info("Extracting timesteps and constructing time mapping")
        times_mjds = tab.getcol("TIME_CENTROID")

        # get unique time steps and make sure they are in their first appeared order
        times_mjds, indices = np.unique(times_mjds, return_index=True)
        sorted_idx = np.argsort(indices)
        times_mjds = times_mjds[sorted_idx]
        time_map = {k: idx for idx, k in enumerate(times_mjds)}

    if whole_day:
        logger.info(f"Assuming a full day from {times_mjds} MJD (seconds)")
        time_step = times_mjds[1] - times_mjds[0]
        times_mjds = times_mjds[0] + time_step * np.arange(
            int(60 * 60 * 24 / time_step)
        )

    times = Time(times_mjds / 60 / 60 / 24, format="mjd")

    sky_position: SkyCoord = _process_position(
        position=position, times=times, ms_path=ms_path
    )

    lst = times.sidereal_time("apparent", longitude=location.lon)
    hour_angle = lst - sky_position.ra
    mask = hour_angle > 12 * u.hourangle
    hour_angle[mask] -= 24 * u.hourangle

    logger.info("Creatring elevation curve")
    sin_alt = np.arcsin(
        np.sin(location.lat) * np.sin(sky_position[0].dec.rad)
        + np.cos(location.lat) * np.cos(sky_position.dec.rad) * np.cos(hour_angle)
    ).to(u.rad)

    return PositionHourAngles(
        hour_angle=hour_angle,
        time_mjds=times_mjds,
        location=location,
        position=sky_position,
        elevation=sin_alt,
        time=times,
        time_map=time_map,
    )
