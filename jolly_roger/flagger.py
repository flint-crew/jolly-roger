"""Flagging utility for a MS"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from jolly_roger.baselines import get_baselines_from_ms
from jolly_roger.hour_angles import make_hour_angles_for_ms
from jolly_roger.logging import logger
from jolly_roger.uvws import uvw_flagger, xyz_to_uvw


@dataclass
class FlagOptions:
    """Specifications of the flagging to carry out"""

    min_scale_deg: float
    """Minimum angular scale to project to UVW"""
    max_scale_deg: float
    """Maximum angular scale to project to UVW"""


def flag(ms_path: Path, flag_options: FlagOptions) -> Path:
    # Trust no one
    logger.debug(f"{flag_options=}")

    ms_path = Path(ms_path)
    logger.info(f"Flagging {ms_path=}")

    baselines = get_baselines_from_ms(ms_path=ms_path)
    hour_angles = make_hour_angles_for_ms(ms_path=ms_path, position="sun")

    uvws = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles)
    ms_path = uvw_flagger(computed_uvws=uvws)
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
        "--max-scale-deg",
        type=float,
        default=0.5,
        help="The minimum scale required for flagging",
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    flag_options = FlagOptions(
        min_scale_deg=args.min_scale_deg,
        max_scale_deg=args.max_scale_deg,
    )

    flag(ms_path=args.ms_path, flag_options=flag_options)


if __name__ == "__main__":
    cli()
