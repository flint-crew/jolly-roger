"""Utilities to construct properties that scale over hour angle"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from numpy.typing import NDArray

from jolly_roger.baselines import OpenMSTables, get_open_ms_tables
from jolly_roger.logging import logger


@dataclass
class PositionHourAngles:
    """Represent time, hour angles and other quantities for some
    assumed sky position. Time intervals are intended to represent
    those stored in a measurement set."""

    hour_angle: u.rad
    """The hour angle across sampled time intervales of a source for a Earth location"""
    time_mjds: u.Quantity
    """The MJD time in seconds from which other quantities are evalauted against. Should be drawn from a measurement set."""
    location: EarthLocation
    """The location these quantities have been derived from."""
    position: SkyCoord
    """The sky-position that is being used to calculate quantities towards"""
    elevation: u.Quantity
    """The elevation of the ``position` direction across time"""
    time: Time
    """Representation of the `time_mjds` attribute"""
    time_map: dict[float, int]
    """Index mapping of time steps described in `time_mjds` to an array index position.
    This is done by selecting all unique `time_mjds` and ordering in this 'first seen'
    position"""


def resolve_position(position: SkyCoord | str, times: Time) -> SkyCoord:
    """Resolve a named or "sun" position to a SkyCoord, or pass a SkyCoord through.

    Args:
        position (SkyCoord | str): The position, or a name/"sun" to look up
        times (Time): Times used when resolving a time-dependent position (the Sun)

    Returns:
        SkyCoord: The resolved position
    """
    if isinstance(position, str):
        if position.lower() == "sun":
            logger.info("Getting sky-position of the Sun")
            return get_sun(times)
        logger.info(f"Getting sky-position of {position=}")
        return SkyCoord.from_name(position)

    return position


def _get_average_location(locations: EarthLocation) -> EarthLocation:
    centroid = EarthLocation.from_geocentric(
        x=locations.x.mean(),
        y=locations.y.mean(),
        z=locations.z.mean(),
    )

    lon, lat, _ = centroid.to_geodetic()
    height = locations.to_geodetic().height.mean()
    return EarthLocation.from_geodetic(
        lon=lon,
        lat=lat,
        height=height,
    )


def get_location(positions: NDArray[np.floating[Any]]) -> EarthLocation:
    """Array-averaged geocentric location from antenna positions.

    Args:
        positions (NDArray[np.floating[Any]]): Antenna geocentric (X,Y,Z), in m

    Returns:
        EarthLocation: The averaged array location
    """
    locations = EarthLocation.from_geocentric(*positions.T, unit=u.m)
    return _get_average_location(locations)


def get_location_from_tables(open_ms_tables: OpenMSTables) -> EarthLocation:
    return get_location(open_ms_tables.antenna_table.getcol("POSITION"))


def get_location_from_ms(ms_path: Path) -> EarthLocation:
    with get_open_ms_tables(ms_path) as open_ms_tables:
        return get_location_from_tables(open_ms_tables)


def make_hour_angles(
    times_mjds: u.Quantity,
    location: EarthLocation,
    position: SkyCoord | str,
    whole_day: bool = False,
) -> PositionHourAngles:
    """Calculate hour-angle and time quantities for a position at a location.

    Args:
        times_mjds (u.Quantity): The TIME_CENTROID values (all rows), in seconds
        location (EarthLocation): The array location
        position (SkyCoord | str): The sky-direction, or a name/"sun" to resolve
        whole_day (bool, optional): Calculate over a 24 hour period from the first time step. Defaults to False.

    Returns:
        PositionHourAngles: Computed hour angles, normalised times and elevation
    """
    logger.info("Extracting timesteps and constructing time mapping")

    # get unique time steps and make sure they are in their first appeared order
    times_mjds, indices = np.unique(times_mjds, return_index=True)
    sorted_idx = np.argsort(indices)
    times_mjds = times_mjds[sorted_idx]
    time_map = {k: idx for idx, k in enumerate(times_mjds)}

    if whole_day:
        logger.info(f"Assuming a full day from {times_mjds} MJD (seconds)")
        time_step = times_mjds[1] - times_mjds[0]
        times_mjds = np.arange(
            start=times_mjds[0],
            stop=times_mjds[0] + 24 * u.hour,
            step=time_step,
        )

    times = Time(times_mjds, format="mjd", scale="utc")

    sky_position = resolve_position(position=position, times=times)

    lst = times.sidereal_time("apparent", longitude=location.lon)
    hour_angle = (lst - sky_position.ra).wrap_at(12 * u.hourangle)

    logger.info("Creating elevation curve")
    altaz = sky_position.transform_to(AltAz(obstime=times, location=location))

    return PositionHourAngles(
        hour_angle=hour_angle,
        time_mjds=cast(u.Quantity, times_mjds),
        location=location,
        position=sky_position,
        elevation=altaz.alt.to(u.rad),
        time=times,
        time_map=time_map,
    )


def make_hour_angles_from_tables(
    open_ms_tables: OpenMSTables,
    position: SkyCoord | str | None = None,
    whole_day: bool = False,
) -> PositionHourAngles:
    """Compute hour angles from open MS tables.

    If ``position`` is None the phase direction of the MS is used.

    Args:
        open_ms_tables (OpenMSTables): The open MS tables to read from
        position (SkyCoord | str | None, optional): The sky-direction. Defaults to None.
        whole_day (bool, optional): Calculate over a full day. Defaults to False.

    Returns:
        PositionHourAngles: Computed hour angles, normalised times and elevation
    """
    times_mjds = open_ms_tables.main_table.getcol("TIME_CENTROID")[:] * u.s
    location = get_location_from_tables(open_ms_tables)

    if position is None:
        position = open_ms_tables.phase_dir

    return make_hour_angles(
        times_mjds=times_mjds,
        location=location,
        position=position,
        whole_day=whole_day,
    )


def make_hour_angles_for_ms(
    ms_path: Path,
    position: SkyCoord | str | None = None,
    whole_day: bool = False,
) -> PositionHourAngles:
    """Calculate hour-angle and time quantities for a given position using time information
    encoded in a nominated measurement set at a nominated location

    Args:
        ms_path (Path): Measurement set to usefor time and sky-position information
        position (SkyCoord | str | None, optional): The sky-direction hour-angles will be calculated towards. Defaults to None.
        whole_day (bool, optional): Calaculate for a 24 hour persion starting from the first time step. Defaults to False.

    Returns:
        PositionHourAngle: Compute hour angles, normalised times and elevation
    """
    logger.info(f"Computing hour angles for {ms_path=}")
    with get_open_ms_tables(ms_path) as open_ms_tables:
        return make_hour_angles_from_tables(
            open_ms_tables=open_ms_tables,
            position=position,
            whole_day=whole_day,
        )
