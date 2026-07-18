"""Routines and structures to describe antennas, their
XYZ and baseline vectors"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c as speed_of_light
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
    antenna_table: table
    """The antenna table"""
    ms_path: Path
    """The path to the MS used to open tables"""
    phase_dir: SkyCoord
    """The phase direction appropriate for the data"""
    nominal_fov: u.Quantity
    """The FWHM field-of-view at the lowest frequency representative of the MS"""

    def close(self) -> None:
        """Close all open tables"""
        for tab in (
            self.main_table,
            self.spw_table,
            self.field_table,
            self.antenna_table,
        ):
            tab.close()

    def __enter__(self) -> OpenMSTables:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def get_phase_dir(
    field_id: NDArray[np.int_],
    phase_dir: NDArray[np.floating[Any]],
) -> SkyCoord:
    """Phase direction for a single-field data set.

    Args:
        field_id (NDArray[np.int_]): The FIELD_ID column (all rows)
        phase_dir (NDArray[np.floating[Any]]): The FIELD table PHASE_DIR column

    Raises:
        ValueError: Raised when more than one FIELD_ID is present

    Returns:
        SkyCoord: The phase direction of the field
    """
    unique_field_id = np.unique(field_id)
    if len(unique_field_id) > 1:
        msg = f"Expected a single FIELD_ID, found: {unique_field_id=}"
        raise ValueError(msg)

    sky_pos = phase_dir[unique_field_id[0]]
    return SkyCoord(*(sky_pos).squeeze() * u.rad)


def _get_phase_dir_for_field_id(ms_table: table, field_table: table) -> SkyCoord:
    """Read FIELD_ID/PHASE_DIR from open tables and derive the phase direction.

    Args:
        ms_table (table): The opened main data table in an MS
        field_table (table): The corresponding opened FIELD table

    Returns:
        SkyCoord: The phase direction of the field
    """
    return get_phase_dir(
        field_id=ms_table.getcol("FIELD_ID"),
        phase_dir=field_table.getcol("PHASE_DIR"),
    )


def beam_fraction_to_radius(
    fraction: float,
    field_of_view: u.Quantity,
) -> u.Quantity:
    """Calculate the angular scale of the radius from the center (e.g. pointed direction)
    out to a elected antennuation level of the Gaussian primary beam
    response.

    A lower fraction leds to a larger field of view, and hence a larger guarding band around
    delay of 0.

    Args:
        fraction (float): The fraction to calculate the distance to
        field_of_view (u.Quantity): The FWHM of the field of view

    Returns:
        u.Quantity: The adius size in radians
    """
    if not 0.0 < fraction < 1.0:
        msg = f"{fraction=} but needs to be in the range (0, 1)."
        raise ValueError(msg)

    beam_fwhm_rad = field_of_view.to("rad").value
    beam_sigma_rad = beam_fwhm_rad / (2 * np.sqrt(2 * np.log(2)))
    sigma_request = np.sqrt(2 * np.log(1 / fraction))
    requested_fov_rad = beam_sigma_rad * sigma_request

    logger.info(f"Request attenuation level: {fraction:.2f}")
    logger.info(f"Requested FoV (radius): {np.rad2deg(requested_fov_rad):.2f} degrees")

    return requested_fov_rad * u.rad


def get_nominal_fov(
    chan_freq: NDArray[np.floating[Any]],
    dish_diameter: NDArray[np.floating[Any]],
) -> u.Quantity:
    """Field of view at the lowest frequency, assuming a single SPW and dish size.

    Args:
        chan_freq (NDArray[np.floating[Any]]): The CHAN_FREQ column, in Hz
        dish_diameter (NDArray[np.floating[Any]]): The DISH_DIAMETER column, in m

    Returns:
        u.Quantity: The nominal radial field-of-view
    """
    # NOTE; This crew member recognises that we could perhaps also convert
    # the FWHM -> radius to some PB level here. However, this is called by
    # the open ms table interface, and I would rather not change that entry
    # point for an optional/opt-in method

    lowest_freq = np.min(chan_freq) * u.Hz
    longest_lambda = (speed_of_light / lowest_freq).decompose()

    unique_diameter = np.unique(dish_diameter)
    assert len(unique_diameter) == 1, (
        f"{len(unique_diameter)} dish sizes found, which is not reasonable"
    )

    fov = (1.02 * longest_lambda / (unique_diameter[0] * u.m)).decompose() * u.rad
    logger.info(f"Nominal field-of-view (FWHM) is {fov.to('deg'):.3f}")

    return fov


def _get_nominal_fov(
    spw_table: table,
    antenna_table: table,
) -> u.Quantity:
    """Read CHAN_FREQ/DISH_DIAMETER from open tables and derive the nominal FoV.

    Args:
        spw_table (table): The spectral window table of the MS
        antenna_table (table): The antenna table of the MS

    Returns:
        u.Quantity: The nominal radial field-of-view
    """
    return get_nominal_fov(
        chan_freq=spw_table.getcol("CHAN_FREQ"),
        dish_diameter=antenna_table.getcol("DISH_DIAMETER"),
    )


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
    antenna_table = table(str(ms_path / "ANTENNA"), ack=False, readonly=read_only)

    phase_dir = _get_phase_dir_for_field_id(
        ms_table=main_table, field_table=field_table
    )
    nominal_fov = _get_nominal_fov(spw_table=spw_table, antenna_table=antenna_table)
    # TODO: Get the data without auto-correlations e.g.
    # no_auto_main_table = taql(
    #     "select from $main_table where ANTENNA1 != ANTENNA2",
    # )

    return OpenMSTables(
        main_table=main_table,
        spw_table=spw_table,
        field_table=field_table,
        antenna_table=antenna_table,
        ms_path=ms_path,
        phase_dir=phase_dir,
        nominal_fov=nominal_fov,
    )


@dataclass
class BaselineData:
    """Container for baseline data and associated metadata."""

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
    ms_path: Path | None = None
    """MS from which the data was fetched, if any"""


@dataclass
class BaselineArrays:
    data: NDArray[np.complexfloating[Any]]
    flags: NDArray[np.bool_]
    uvws: NDArray[np.floating[Any]]
    time_centroid: NDArray[np.floating[Any]]


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


def build_baseline_data(
    baseline_arrays: BaselineArrays,
    freq_chan: NDArray[np.floating[Any]],
    phase_dir: SkyCoord,
    ant_1: int,
    ant_2: int,
    ms_path: Path | None = None,
) -> BaselineData:
    """Attach units and masking to raw baseline arrays.

    Args:
        baseline_arrays (BaselineArrays): The raw columns for a baseline
        freq_chan (NDArray[np.floating[Any]]): The CHAN_FREQ column, in Hz
        phase_dir (SkyCoord): The phase direction of the data
        ant_1 (int): The first antenna of the baseline
        ant_2 (int): The second antenna of the baseline
        ms_path (Path | None, optional): The source MS, if any. Defaults to None.

    Returns:
        BaselineData: The unit-attached baseline data
    """
    uvws_phase_center = np.swapaxes(baseline_arrays.uvws * u.m, 0, 1)
    time = Time(
        baseline_arrays.time_centroid.squeeze() * u.s,
        format="mjd",
        scale="utc",
    )
    masked_data = np.ma.masked_array(baseline_arrays.data, mask=baseline_arrays.flags)

    return BaselineData(
        masked_data=masked_data,
        freq_chan=freq_chan.squeeze() * u.Hz,
        phase_center=phase_dir,
        uvws_phase_center=cast(u.Quantity, uvws_phase_center),
        time=time,
        ant_1=ant_1,
        ant_2=ant_2,
        ms_path=ms_path,
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

    baseline_arrays = _get_baseline_data(
        ms_tab=open_ms_tables.main_table,
        ant_1=ant_1,
        ant_2=ant_2,
        data_column=data_column,
    )

    baseline_data = build_baseline_data(
        baseline_arrays=baseline_arrays,
        freq_chan=open_ms_tables.spw_table.getcol("CHAN_FREQ"),
        phase_dir=open_ms_tables.phase_dir,
        ant_1=ant_1,
        ant_2=ant_2,
        ms_path=open_ms_tables.ms_path,
    )

    logger.info(
        f"Got data for baseline {ant_1} {ant_2} with shape {baseline_data.masked_data.shape}"
    )
    return baseline_data


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
    ms_path: Path | None = None
    """The measurement set used to construct some instance of `Baseline`, if any"""


def get_baselines(
    ant_xyz: NDArray[np.floating[Any]],
    reverse_baselines: bool = False,
    ms_path: Path | None = None,
) -> Baselines:
    """Form the upper-triangle baseline vectors from antenna positions.

    Args:
        ant_xyz (NDArray[np.floating[Any]]): Antenna (X,Y,Z) positions, in m
        reverse_baselines (bool, optional): Reverse the baseline ordering. Defaults to False.
        ms_path (Path | None, optional): The source MS, if any. Defaults to None.

    Returns:
        Baselines: The corresponding set of baselines formed.
    """
    ants_idx = np.arange(len(ant_xyz), dtype=int)
    b_idx = np.array(list(combinations(list(ants_idx), 2)))
    if reverse_baselines:
        b_idx = b_idx[:, ::-1]
    b_xyz = ant_xyz[b_idx[:, 0]] - ant_xyz[b_idx[:, 1]]

    b_map = {tuple(k): idx for idx, k in enumerate(b_idx)}

    logger.info(f"ants={len(ants_idx)}, baselines={b_idx.shape[0]}")
    return Baselines(
        ant_xyz=ant_xyz * u.m,
        b_xyz=b_xyz * u.m,
        b_idx=b_idx,
        b_map=b_map,
        ms_path=ms_path,
    )


def get_baselines_from_tables(
    open_ms_tables: OpenMSTables,
    reverse_baselines: bool = False,
) -> Baselines:
    """Form baselines from the ANTENNA positions of open MS tables.

    Args:
        open_ms_tables (OpenMSTables): The open MS tables to read from
        reverse_baselines (bool): Reverse the baseline ordering

    Returns:
        Baselines: The corresponding set of baselines formed.
    """
    return get_baselines(
        ant_xyz=open_ms_tables.antenna_table.getcol("POSITION"),
        reverse_baselines=reverse_baselines,
        ms_path=open_ms_tables.ms_path,
    )


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
    with get_open_ms_tables(ms_path) as open_ms_tables:
        return get_baselines_from_tables(
            open_ms_tables=open_ms_tables,
            reverse_baselines=reverse_baselines,
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

    assert baselines.ms_path is not None, "baselines has no ms_path"
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
