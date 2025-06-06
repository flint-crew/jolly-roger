"""Calculating the UVWs for a measurement set towards a direction"""

from __future__ import annotations

import astropy.units as u
import numpy as np

from jolly_roger.hour_angles import PositionHourAngle
from jolly_roger.logging import logger


def all_b_xyz_to_uvw(
    b_xyz: np.ndarray,
    hour_angle: PositionHourAngle,
):
    # Getting the units right is important, mate
    ha = hour_angle.hour_angle
    ha = ha.to(u.rad)

    declination = hour_angle.position.declination
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
    uvw = np.einsum("ijk,lj->lik", mat, b_xyz, optimize=True)  # qa: ignore

    logger.debug(f"{uvw.shape=}")

    return uvw
