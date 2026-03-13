"""Routines and structures to describe antennas, their
XYZ and baseline vectors"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import cast

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from casacore.tables import table, taql
from numpy.typing import NDArray

from jolly_roger.logging import logger


@dataclass(frozen=True)
class OpenMSTables:
    """Open MS table references"""

    main_table: table
    """The main MS table"""
    spw_table: table
    """The spectral window table"""
    field_table: table
    """The field table"""
    ms_path: Path
    """The path to the MS used to open tables"""


def get_open_ms_tables(ms_path: Path, read_only: bool = True) -> OpenMSTables:
    """Open up the set of MS table and sub-tables necessary for tractoring.

    Args:
        ms_path (Path): The path to the measurement set
        read_only (bool, optional): Whether to open in a read-only mode. Defaults to True.

    Returns:
        OpenMSTables: Set of open table references
    """
    main_table = table(str(ms_path), ack=False, readonly=read_only)
    spw_table = table(str(ms_path / "SPECTRAL_WINDOW"), ack=False, readonly=read_only)
    field_table = table(str(ms_path / "FIELD"), ack=False, readonly=read_only)

    # TODO: Get the data without auto-correlations e.g.
    # no_auto_main_table = taql(
    #     "select from $main_table where ANTENNA1 != ANTENNA2",
    # )

    return OpenMSTables(
        main_table=main_table,
        spw_table=spw_table,
        field_table=field_table,
        ms_path=ms_path,
    )


@dataclass
class BaselineData:
    """Container for baseline data and associated metadata."""

    ms_path: Path
    """MS from which the data was fetched"""
    masked_data: np.ma.MaskedArray
    """The baseline data, masked where flags are set. shape=(time, chan, pol)"""
    freq_chan: u.Quantity
    """The frequency channels corresponding to the data."""
    phase_center: SkyCoord
    """The target sky coordinate for the baseline."""
    uvws_phase_center: u.Quantity
    """The UVW coordinates of the phase center of the baseline."""
    time: Time
    """The time of the observations."""
    ant_1: int
    """The first antenna in the baseline."""
    ant_2: int
    """The second antenna in the baseline."""


@dataclass
class BaselineArrays:
    data: NDArray[np.complexfloating]
    flags: NDArray[np.bool_]
    uvws: NDArray[np.floating]
    time_centroid: NDArray[np.floating]


def _get_baseline_data(
    ms_tab: table,
    ant_1: int,
    ant_2: int,
    data_column: str = "DATA",
) -> BaselineArrays:
    _ = ms_tab, ant_1, ant_2
    with taql(
        "select from $ms_tab where ANTENNA1 == $ant_1 and ANTENNA2 == $ant_2",
    ) as subtab:
        logger.info(f"Opening subtable for baseline {ant_1} {ant_2}")
        data = subtab.getcol(data_column)
        flags = subtab.getcol("FLAG")
        uvws = subtab.getcol("UVW")
        time_centroid = subtab.getcol("TIME_CENTROID")

    return BaselineArrays(
        data=data,
        flags=flags,
        uvws=uvws,
        time_centroid=time_centroid,
    )


def get_baseline_data(
    open_ms_tables: OpenMSTables,
    ant_1: int,
    ant_2: int,
    data_column: str = "DATA",
) -> BaselineData:
    """Get data of a baseline from a measurement set

    Args:
        open_ms_tables (OpenMSTables): The measurement set to draw data from
        ant_1 (int): The first antenna of the baseline
        ant_2 (int): The second antenna of the baseline
        data_column (str, optional): The data column to extract. Defaults to "DATA".

    Returns:
        BaselineData:  Extracted baseline data
    """
    logger.info(f"Getting baseline {ant_1} {ant_2}")

    freq_chan = open_ms_tables.spw_table.getcol("CHAN_FREQ")
    phase_dir = open_ms_tables.field_table.getcol("PHASE_DIR")

    logger.debug(f"Processing {ant_1=} {ant_2=}")

    baseline_data = _get_baseline_data(
        ms_tab=open_ms_tables.main_table,
        ant_1=ant_1,
        ant_2=ant_2,
        data_column=data_column,
    )

    freq_chan = freq_chan.squeeze() * u.Hz
    target = SkyCoord(*(phase_dir * u.rad).squeeze())
    uvws_phase_center = np.swapaxes(baseline_data.uvws * u.m, 0, 1)
    time = Time(
        baseline_data.time_centroid.squeeze() * u.s,
        format="mjd",
        scale="utc",
    )
    masked_data = np.ma.masked_array(baseline_data.data, mask=baseline_data.flags)

    logger.info(f"Got data for baseline {ant_1} {ant_2} with shape {masked_data.shape}")
    return BaselineData(
        ms_path=open_ms_tables.ms_path,
        masked_data=masked_data,
        freq_chan=freq_chan,
        phase_center=target,
        uvws_phase_center=cast(u.Quantity, uvws_phase_center),
        time=time,
        ant_1=ant_1,
        ant_2=ant_2,
    )


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


def get_baselines_from_ms(
    ms_path: Path,
    reverse_baselines: bool = False,
) -> Baselines:
    """Extract the antenna positions from the nominated measurement
    set and constructed the set of baselines. These are drawn from
    the ANTENNA table in the measurement set.

    Args:
        ms_path (Path): The measurement set to extract baseliens from
        reverse_baselines (bool): Reverse the baseline ordering

    Returns:
        Baselines: The corresponding set of baselines formed.
    """

    logger.info(f"Creating baseline instance from {ms_path=}")
    with table(str(ms_path / "ANTENNA"), ack=False) as tab:
        ants_idx = np.arange(len(tab), dtype=int)
        b_idx = np.array(list(combinations(list(ants_idx), 2)))
        if reverse_baselines:
            b_idx = b_idx[:, ::-1]
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
