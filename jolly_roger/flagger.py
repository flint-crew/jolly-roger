"""Flagging utility for a MS"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u

from jolly_roger.baselines import get_baselines_from_ms
from jolly_roger.hour_angles import make_hour_angles_for_ms
from jolly_roger.logging import logger
from jolly_roger.uvws import uvw_flagger, xyz_to_uvw


@dataclass
class JollyRogerFlagOptions:
    """Specifications of the flagging to carry out"""

    min_scale_deg: float = 0.075
    """Minimum angular scale to project to UVW"""
    min_horizon_limit_deg: float = -3
    """The minimum elevation for the sun projected baselines to be considered for flagging"""
    max_horizon_limit_deg: float = 90
    """The minimum elevation for the sun projected baselines to be considered for flagging"""
    dry_run: bool = False
    """Do not apply the flags"""


def flag(ms_path: Path, flag_options: JollyRogerFlagOptions) -> Path:
    # Trust no one
    logger.debug(f"{flag_options=}")

    ms_path = Path(ms_path)
    logger.info(f"Flagging {ms_path=}")

    baselines = get_baselines_from_ms(ms_path=ms_path)
    hour_angles = make_hour_angles_for_ms(ms_path=ms_path, position="sun")

    uvws = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles)
    ms_path = uvw_flagger(
        computed_uvws=uvws,
        min_horizon_lim=flag_options.min_horizon_limit_deg * u.deg,
        max_horizon_lim=flag_options.max_horizon_limit_deg * u.deg,
        min_sun_scale=flag_options.min_scale_deg * u.deg,
        dry_run=flag_options.dry_run,
    )
    logger.info(f"Finished processing {ms_path=}")

    return ms_path


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Flag a measurement set based on properties of the Sun"
    )
    parser.add_argument("ms_path", type=Path, help="The measurement set to flag")

    parser.add_argument(
        "--min-scale-deg",
        type=float,
        default=0.075,
        help="The minimum scale required for flagging",
    )
    parser.add_argument(
        "--min-horizon-limit-deg",
        type=float,
        default=-3,
        help="The minimum elevation of the centroid of the object (e.g. sun) for uvw flagging to be activated",
    )
    parser.add_argument(
        "--max-horizon-limit-deg",
        type=float,
        default=-3,
        help="The maximum elevation of the centroid of the object (e.g. sun) for uvw flagging to be activated",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not apply the computed flags"
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    flag_options = JollyRogerFlagOptions(
        min_scale_deg=args.min_scale_deg,
        min_horizon_limit_deg=args.min_horizon_limit_deg,
        max_horizon_limit_deg=args.max_horizon_limit_deg,
        dry_run=args.dry_run,
    )

    flag(ms_path=args.ms_path, flag_options=flag_options)


if __name__ == "__main__":
    cli()
