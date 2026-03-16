"""Flagging utility for a MS"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import astropy.units as u
from capn_crunch import BaseOptions, add_options_to_parser, create_options_from_parser

from jolly_roger.baselines import get_baselines_from_ms
from jolly_roger.hour_angles import make_hour_angles_for_ms
from jolly_roger.logging import logger
from jolly_roger.uvws import uvw_flagger, xyz_to_uvw


class JollyRogerFlagOptions(BaseOptions):
    """Specifications of the flagging to carry out"""

    min_scale_deg: float = 0.075
    """Minimum angular scale to project to UVW"""
    min_horizon_limit_deg: float = -3
    """The minimum elevation for the sun projected baselines to be considered for flagging"""
    max_horizon_limit_deg: float = 90
    """The minimum elevation for the sun projected baselines to be considered for flagging"""
    dry_run: bool = False
    """Do not apply the flags"""
    flip_uvw_sign: bool = False
    """Flip the sign of UVWs (may be required for LOFAR)"""


def flag(ms_path: Path, flag_options: JollyRogerFlagOptions) -> Path:
    # Trust no one
    logger.debug(f"{flag_options=}")

    ms_path = Path(ms_path)
    logger.info(f"Flagging {ms_path=}")

    baselines = get_baselines_from_ms(ms_path=ms_path)
    hour_angles = make_hour_angles_for_ms(ms_path=ms_path, position="sun")

    uvws = xyz_to_uvw(
        baselines=baselines,
        hour_angles=hour_angles,
        flip_uvw_sign=flag_options.flip_uvw_sign,
    )
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
        description="Flag a measurement set based on properties of the Sun",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ms_path", type=Path, help="The measurement set to flag")

    return add_options_to_parser(
        parser=parser,
        options_class=JollyRogerFlagOptions,
    )


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    flag_options = create_options_from_parser(
        parser_namespace=args,
        options_class=JollyRogerFlagOptions,
    )

    flag(ms_path=args.ms_path, flag_options=flag_options)


if __name__ == "__main__":
    cli()
