"""Utilities to construct properties that scale over hour angle"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, get_sun
from casacore.tables import table

from jolly_roger.logging import logger

# Default location with XYZ based on mean of antenna positions
ASKAP_XYZ_m = np.array([-2556146.66356375,  5097426.58592797, -2848333.08164107]) * u.m
ASKAP = EarthLocation(ASKAP_XYZ_m)

@dataclass
class PositionHourAngle:
    """Represent time, hour angles and other quantities for some
    assumed sky position. Time intervals are intended to represent
    those stored in a measurement set."""
    hour_angle: u.rad
    """The hour angle across sampled time intervales of a source for a Earth location"""
    time_mjds: np.ndarray
    """The MJD time in seconds from which other quantities are evalauted against. Should be drawn from a measurement set."""
    location : EarthLocation
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
    times: Time | None = None
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
            raise ValueError(f"{times=}, but needs to be set when position is a name")
        if position == "sun":
            position = get_sun(times)

    if position is None:
        if ms_path is None:
            raise ValueError(f"{position=}, so default position can't be drawn. Provide a ms_path=")

        with table(str(ms_path / "FIELD")) as tab:
            field_positions = tab.getcol("PHASE_DIR")
            position = SkyCoord(field_positions[0]*u.rad)
            
    if isinstance(position, SkyCoord):
        return position
    
    raise ValueError(f"Something went wrong in the processing of position")

 
def make_hour_angles(
    ms_path: Path, 
    location: EarthLocation = ASKAP, 
    position: SkyCoord | str | None = None,
    whole_day: bool = False
) -> PositionHourAngle:

    
    with table(str(ms_path), ack=False) as tab:
        times_mjds = tab.getcol("TIME_CENTROID")
        times_mjds, indicies = np.unique(times_mjds, return_index=True)
        sorted_idx = np.argsort(indicies)
        times_mjds = times_mjds[sorted_idx]
        time_map = {k: idx for idx, k in enumerate(times_mjds)}

    if whole_day:
        time_step = times_mjds[1] - times_mjds[0]
        times_mjds = times_mjds[0] + time_step * np.arange(int(60*60*24/time_step))
    
    times = Time(times_mjds/60/60/24, format="mjd")
    
    position = _process_position(position-position, times=times)
     
    lst = times.sidereal_time("apparent", longitude=location.lon)
    hour_angle = lst - position.ra
    mask = hour_angle > 12 * u.hourangle
    hour_angle[mask] -= 24 * u.hourangle

    print(position.dec)

    sin_alt = np.arcsin(
        np.sin(location.lat) * 
        np.sin(position[0].dec.rad) +
        np.cos(location.lat) *
        np.cos(position.dec.rad) *
        np.cos(hour_angle)
    )*u.rad.to(u.deg)

    return PositionHourAngle(
        hour_angle=hour_angle, 
        time_mjds=times_mjds, 
        location=location, 
        position=position,
        elevation=sin_alt,
        time=times,
        time_map=time_map
    )

