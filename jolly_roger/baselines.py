"""Routines and structures to describe antennas, their
XYZ and baseline vectors"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from casacore.tables import table

from jolly_roger.logging import logger


@dataclass
class Baselines:
    """Container representing the antennas found in some measurement set, their
    baselines and associated mappings. Only the upper triangle of
    baselines are formed, e.g. 1-2 not 2-1.
    """

    ant_xyz: np.ndarray
    """Antenna (X,Y,Z) coordinates taken from the measurement set"""
    b_xyz: np.ndarray
    """The baseline vectors formed from each antenna-pair"""
    b_idx: np.ndarray
    """Baselihe indices representing a pair of antenna"""
    b_map: dict[tuple[int, int], int]
    """A mapping between two antennas to their baseline index"""
    ms_path: Path
    """The measurement set used to construct some instance of `Baseline`"""


def get_baselines_from_ms(ms_path: Path) -> Baselines:
    """Extract the antenna positions from the nominated measurement
    set and constructed the set of baselines. These are drawn from
    the ANTENNA table in the measurement set.

    Args:
        ms_path (Path): The measurement set to extract baseliens from

    Returns:
        Baselines: The corresponding set of baselines formed.
    """

    logger.info(f"Creating baseline instance from {ms_path=}")
    with table(str(ms_path / "ANTENNA"), ack=False) as tab:
        ants_idx = np.arange(len(tab), dtype=int)
        b_idx = np.array(list(combinations(list(ants_idx), 2)))
        xyz = tab.getcol("POSITION")
        b_xyz = xyz[b_idx[:, 0]] - xyz[b_idx[:, 1]]

    b_map = {tuple(k): idx for idx, k in enumerate(b_idx)}

    logger.info(f"ants={len(ants_idx)}, baselines={b_idx.shape[0]}")
    return Baselines(
        ant_xyz=xyz * u.m, b_xyz=b_xyz * u.m, b_idx=b_idx, b_map=b_map, ms_path=ms_path
    )


@dataclass
class BaselinePlotPaths:
    """Names for plots for baseline visualisations"""

    antenna_path: Path
    """Output for the antenna XYZ plot"""
    baseline_path: Path
    """Output for the baselines vector plot"""


def make_plot_names(ms_path: Path) -> BaselinePlotPaths:
    """Construct the output paths of the diagnostic plots

    Args:
        ms_path (Path): The measurement set the plots are created for

    Returns:
        BaselinePlotPaths: The output paths for the plot names
    """

    basename = ms_path.parent / ms_path.stem

    antenna_path = Path(f"{basename!s}-antenna.pdf")
    baseline_path = Path(f"{basename!s}-baseline.pdf")

    return BaselinePlotPaths(antenna_path=antenna_path, baseline_path=baseline_path)


def plot_baselines(baselines: Baselines) -> BaselinePlotPaths:
    """Create basic diagnostic plots for a set of baselines. This
    includes the antenna positions and the baseline vectors.

    Args:
        baselines (Baselines): The loaded instance of the baselines from a measurement set

    Returns:
        BaselinePlotPaths: The output paths of the plots created
    """

    plot_names = make_plot_names(ms_path=baselines.ms_path)

    # Make the initial antenna plot
    fig, ax = plt.subplots(1, 1)

    ax.scatter(baselines.b_xyz[:, 0], baselines.b_xyz[:, 1], label="Baseline")

    ax.set(xlabel="X (meters)", ylabel="Y (meters)", title="ASKAP Baseline Vectors")
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_names.baseline_path)

    # Now plot the antennas
    fig, ax = plt.subplots(1, 1)

    ax.scatter(baselines.ant_xyz[:, 0], baselines.ant_xyz[:, 1], label="Antenna")

    ax.set(xlabel="X (meters)", ylabel="Y (meters)", title="ASKAP Antenna positions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_names.antenna_path)

    return plot_names


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Extract and plot antenna dna baseline information from a measurement set"
    )

    sub_parsers = parser.add_subparsers(dest="mode")

    plot_parser = sub_parsers.add_parser(
        "plot", description="Basic plots around baselines"
    )
    plot_parser.add_argument("ms_path", type=Path, help="Path to the measurement set")

    return parser


def cli() -> None:
    parser: ArgumentParser = get_parser()

    args = parser.parse_args()

    if args.mode == "plot":
        baselines = get_baselines_from_ms(ms_path=args.ms_path)
        plot_baselines(baselines=baselines)


if __name__ == "__main__":
    cli()
