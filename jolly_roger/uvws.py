"""Calculating the UVWs for a measurement set towards a direction"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np

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


def all_b_xyz_to_uvw(
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

    declination = hour_angles.position.declination
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
