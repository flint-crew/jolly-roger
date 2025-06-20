from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.constants import c as speed_of_light
from astropy.coordinates import (
    SkyCoord,
)
from astropy.time import Time
from casacore.tables import makecoldesc, table, taql
from numpy.typing import NDArray
from tqdm.auto import tqdm

from jolly_roger.baselines import Baselines, get_baselines_from_ms
from jolly_roger.delays import data_to_delay_time, delay_time_to_data
from jolly_roger.hour_angles import PositionHourAngles, make_hour_angles_for_ms
from jolly_roger.logging import logger
from jolly_roger.plots import plot_baseline_comparison_data
from jolly_roger.uvws import UVWs, xyz_to_uvw


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

    return OpenMSTables(
        main_table=main_table,
        spw_table=spw_table,
        field_table=field_table,
        ms_path=ms_path,
    )


def tukey_taper(
    x: np.typing.NDArray[np.floating],
    outer_width: float = np.pi / 4,
    tukey_width: float = np.pi / 8,
    tukey_x_offset: NDArray[np.floating] | None = None,
) -> np.ndarray:
    x_freq = np.linspace(-np.pi, np.pi, len(x))
    taper = np.ones_like(x_freq)

    _ = tukey_x_offset

    # Fully zero region
    taper[np.abs(x_freq) > outer_width] = 0

    # Transition regions
    left_idx = (-outer_width < x_freq) & (x_freq < -outer_width + tukey_width)
    right_idx = (outer_width - tukey_width < x_freq) & (x_freq < outer_width)

    taper[left_idx] = (
        1 - np.cos(np.pi * (x_freq[left_idx] + outer_width) / tukey_width)
    ) / 2

    taper[right_idx] = (
        1 - np.cos(np.pi * (outer_width - x_freq[right_idx]) / tukey_width)
    ) / 2

    return taper


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


@dataclass
class BaselineArrays:
    data: NDArray[np.complexfloating]
    flags: NDArray[np.bool_]
    uvws: NDArray[np.floating]
    time_centroid: NDArray[np.floating]


@dataclass
class DataChunkArray:
    """Container for a chunk of data"""

    data: NDArray[np.complexfloating]
    """The data from the nominated data column loaded"""
    flags: NDArray[np.bool_]
    """Flags that correspond to the loaded data"""
    uvws: NDArray[np.floating]
    """The uvw coordinates for each loaded data record"""
    time_centroid: NDArray[np.floating]
    """The time of each data record"""
    ant_1: NDArray[np.int64]
    """Antenna 1 that formed the baseline"""
    ant_2: NDArray[np.int64]
    """Antenna 2 that formed the baseline"""
    row_start: int
    """The starting row of the portion of data loaded"""
    chunk_size: int
    """The size of the data chunk loaded (may be larger if this is the last record)"""


@dataclass
class DataChunk:
    """Container for a collection of data and associated metadata.
    Here data are drawn from a series of rows.
    """

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
    time_mjds: NDArray[np.floating]
    """The raw time extracted from the measurement set in MJDs"""
    ant_1: NDArray[np.int64]
    """The first antenna in the baseline."""
    ant_2: NDArray[np.int64]
    """The second antenna in the baseline."""
    row_start: int
    """Starting row index of the data"""
    chunk_size: int
    """Size of the chunked portion of the data"""


def _list_to_array(
    list_of_rows: list[dict[str, Any]], key: str
) -> np.typing.NDArray[Any]:
    """Helper to make a simple numpy object from list of items"""
    return np.array([row[key] for row in list_of_rows])


def _get_data_chunk_from_main_table(
    ms_table: table,
    chunk_size: int,
    data_column: str,
) -> Generator[DataChunkArray, None, None]:
    """Return an appropriately size data chunk from the main
    table of a measurement set. These data are ase they are
    in the measurement set without any additional scaling
    or unit adjustments.

    Args:
        ms_table (table): The opened main table of a measurement set
        chunk_size (int): The size of the data to chunk and return
        data_column (str): The data column to be returned

    Yields:
        Generator[DataChunkArray, None, None]: A segment of rows and columns
    """

    table_length = len(ms_table)
    logger.debug(f"Length of open table: {table_length} rows")

    lower_row = 0
    upper_row = chunk_size

    while lower_row < table_length:
        rows: list[dict[str, Any]] = ms_table[lower_row:upper_row]

        data = _list_to_array(list_of_rows=rows, key=data_column)
        flags = _list_to_array(list_of_rows=rows, key="FLAG")
        uvws = _list_to_array(list_of_rows=rows, key="UVW")
        time_centroid = _list_to_array(list_of_rows=rows, key="TIME_CENTROID")
        ant_1 = _list_to_array(list_of_rows=rows, key="ANTENNA1")
        ant_2 = _list_to_array(list_of_rows=rows, key="ANTENNA2")

        yield DataChunkArray(
            data=data,
            flags=flags,
            uvws=uvws,
            time_centroid=time_centroid,
            ant_1=ant_1,
            ant_2=ant_2,
            row_start=lower_row,
            chunk_size=chunk_size,
        )

        lower_row += chunk_size
        upper_row += chunk_size


def get_data_chunks(
    open_ms_tables: OpenMSTables,
    chunk_size: int,
    data_column: str,
) -> Generator[DataChunk, None, None]:
    """Yield a collection of rows with appropriate units
    attached to the quantities. These quantities are not
    the same data encoded in the measurement set, e.g.
    masked array has been formed, astropy units have
    been attached.

    Args:
        open_ms_tables (OpenMSTables): References to open tables from the measurement set
        chunk_size (int): The number of rows to return at a time
        data_column (str): The data column that would be modified

    Yields:
        Generator[DataChunk, None, None]: Representation of the current chunk of rows
    """
    freq_chan = open_ms_tables.spw_table.getcol("CHAN_FREQ")
    phase_dir = open_ms_tables.field_table.getcol("PHASE_DIR")

    freq_chan = freq_chan.squeeze() * u.Hz
    target = SkyCoord(*(phase_dir * u.rad).squeeze())

    for data_chunk_array in _get_data_chunk_from_main_table(
        ms_table=open_ms_tables.main_table,
        chunk_size=chunk_size,
        data_column=data_column,
    ):
        # Transform the native arrays but attach astropy quantities
        uvws_phase_center = data_chunk_array.uvws * u.m
        time = Time(
            data_chunk_array.time_centroid.squeeze() * u.s,
            format="mjd",
            scale="utc",
        )
        masked_data = np.ma.masked_array(
            data_chunk_array.data, mask=data_chunk_array.flags
        )

        yield DataChunk(
            masked_data=masked_data,
            freq_chan=freq_chan,
            phase_center=target,
            uvws_phase_center=uvws_phase_center,
            time=time,
            time_mjds=data_chunk_array.time_centroid,
            ant_1=data_chunk_array.ant_1,
            ant_2=data_chunk_array.ant_2,
            row_start=data_chunk_array.row_start,
            chunk_size=data_chunk_array.chunk_size,
        )


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
        masked_data=masked_data,
        freq_chan=freq_chan,
        phase_center=target,
        uvws_phase_center=uvws_phase_center,
        time=time,
        ant_1=ant_1,
        ant_2=ant_2,
    )


def add_output_column(
    tab: table,
    data_column: str = "DATA",
    output_column: str = "CORRECTED_DATA",
    overwrite: bool = False,
    copy_column_data: bool = False,
) -> None:
    """Add in the output data column where the modified data
    will be recorded

    Args:
        tab (table): Open reference to the table to modify
        data_column (str, optional): The base data column the new will be based from. Defaults to "DATA".
        output_column (str, optional): The new data column to be created. Defaults to "CORRECTED_DATA".
        overwrite (bool, optional): Whether to overwrite the new output column. Defaults to False.
        copy_column_data (bool, optional): Copy the original data over to the output column. Defaults to False.

    Raises:
        ValueError: Raised if the output column already exists and overwrite is False
    """
    colnames = tab.colnames()
    if output_column in colnames:
        if not overwrite:
            msg = f"Output column {output_column} already exists in the measurement set. Not overwriting."
            raise ValueError(msg)

        logger.warning(
            f"Output column {output_column} already exists in the measurement set. Will be overwritten!"
        )
        return
    logger.info(f"Adding {output_column=}")
    desc = makecoldesc(data_column, tab.getcoldesc(data_column))
    desc["name"] = output_column
    tab.addcols(desc)
    tab.flush()
    if copy_column_data:
        logger.info(f"Copying {data_column=} to {output_column=}")
        taql(f"UPDATE $tab SET {output_column}={data_column}")


def write_output_column(
    ms_path: Path,
    output_column: str,
    baseline_data: BaselineData,
    update_flags: bool = False,
) -> None:
    """Write the output column to the measurement set."""
    ant_1 = baseline_data.ant_1
    ant_2 = baseline_data.ant_2
    _ = ant_1, ant_2
    logger.info(f"Writing {output_column=} for baseline {ant_1} {ant_2}")
    with table(str(ms_path), readonly=False) as tab:
        colnames = tab.colnames()
        if output_column not in colnames:
            msg = f"Output column {output_column} does not exist in the measurement set. Cannot write data."
            raise ValueError(msg)

        with taql(
            "select from $tab where ANTENNA1 == $ant_1 and ANTENNA2 == $ant_2",
        ) as subtab:
            logger.info(f"Writing {output_column=}")
            subtab.putcol(output_column, baseline_data.masked_data.filled(0 + 0j))
            if update_flags:
                # If we want to update the flags, we need to set the flags to False
                # for the output column
                subtab.putcol("FLAG", baseline_data.masked_data.mask)
            subtab.flush()


def make_plot_results(
    open_ms_tables: OpenMSTables, data_column: str, output_column: str
) -> list[Path]:
    output_paths = []
    output_dir = open_ms_tables.ms_path.parent / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    for i in range(10):
        logger.info(f"Plotting baseline={i + 1}")
        before_baseline_data = get_baseline_data(
            open_ms_tables=open_ms_tables,
            ant_1=0,
            ant_2=i + 1,
            data_column=data_column,
        )
        after_baseline_data = get_baseline_data(
            open_ms_tables=open_ms_tables,
            ant_1=0,
            ant_2=i + 1,
            data_column=output_column,
        )
        before_delays = data_to_delay_time(data=before_baseline_data)
        after_delays = data_to_delay_time(data=after_baseline_data)

        # TODO: the baseline data and delay times could be put into a single
        # structure to pass around easier.
        plot_path = plot_baseline_comparison_data(
            before_baseline_data=before_baseline_data,
            after_baseline_data=after_baseline_data,
            before_delays=before_delays,
            after_delays=after_delays,
            output_dir=output_dir,
            suffix="_comparison",
        )
        output_paths.append(plot_path)

    return output_paths


def _tukey_tractor(
    data_chunk: DataChunk,
    tukey_tractor_options: TukeyTractorOptions,
    w_delays: WDelays | None = None,
) -> NDArray[np.complex128]:
    """Compute a tukey taper for a dataset and then apply it
    to the dataset. Here the data corresponds to a (chan, time, pol)
    array. Data is not necessarily a single baseline.

    If a `w_delays` is provided it represents the delay (in seconds)
    between the phase direction of the measurement set and the Sun.
    This quantity may be derived in a number of ways, but in `jolly_roger`
    it is based on the difference of the w-coordinated towards these
    two directions. It should have a shape of [baselines, time]

    Args:
        data_chunk (DataChunk): The representation of the data with attached units
        tukey_tractor_options (TukeyTractorOptions): Options for the tukey taper
        w_delays (WDelays | None, optional): The w-derived delays to apply. If None taper is applied to large delays. Defaults to None.

    Returns:
        NDArray[np.complex128]: Scaled complex visibilities
    """
    # Look up the delay offset if requested
    tukey_x_offset: None | NDArray[np.floating] = None
    if w_delays is not None:
        # When computing uvws we have ignored auto-correlations!
        # TODO: Either extend the uvw calculations to include auto-correlations
        # or ignore them during iterations. Certainly the former is the better
        # approach.
        baseline_idx = np.array(
            [
                w_delays.b_map[(int(ant_1), int(ant_2))] if ant_1 != ant_2 else 0
                for ant_1, ant_2 in zip(  # type: ignore[call-overload]
                    data_chunk.ant_1, data_chunk.ant_2, strict=False
                )
            ]
        )

        logger.info(w_delays.time_map.keys())
        time_idx = np.array(
            [w_delays.time_map[time * u.s] for time in data_chunk.time_mjds]
        )
        tukey_x_offset = w_delays.w_delays[baseline_idx, time_idx]

    delay_time = data_to_delay_time(data=data_chunk)

    taper = tukey_taper(
        x=delay_time.delay,
        outer_width=tukey_tractor_options.outer_width,
        tukey_width=tukey_tractor_options.tukey_width,
        tukey_x_offset=tukey_x_offset,
    )

    # Delay-time is a 3D array: (time, delay, pol)
    # Taper is 1D: (delay,)
    tapered_delay_time_data_real = (
        delay_time.delay_time.real * taper[np.newaxis, :, np.newaxis]
    )
    tapered_delay_time_data_imag = (
        delay_time.delay_time.imag * taper[np.newaxis, :, np.newaxis]
    )
    tapered_delay_time_data = (
        tapered_delay_time_data_real + 1j * tapered_delay_time_data_imag
    )
    tapered_delay_time = delay_time
    tapered_delay_time.delay_time = tapered_delay_time_data

    tapered_data = delay_time_to_data(
        delay_time=tapered_delay_time,
        original_data=data_chunk,
    )
    logger.debug(f"{tapered_data.masked_data.shape=} {tapered_data.masked_data.dtype}")

    return tapered_data


@dataclass
class TukeyTractorOptions:
    """Options to describe the tukey taper to apply"""

    ms_path: Path
    """Measurement set to be modified"""
    outer_width: float = np.pi / 4
    """The start of the tapering in frequency space"""
    tukey_width: float = np.pi / 8
    """The width of the tapered region in frequency space"""
    data_column: str = "DATA"
    """The visibility column to modify"""
    output_column: str = "CORRECTED_DATA"
    """The output column to be created with the modified data"""
    dry_run: bool = False
    """Indicates whether the data will be written back to the measurement set"""
    make_plots: bool = False
    """Create a small set of diagnostic plots"""
    overwrite: bool = False
    """If the output column exists it will be overwritten"""
    chunk_size: int = 1000
    """Size of the row-wise chunking iterator"""
    apply_towards_sun: bool = False
    """apply the taper using the delay towards the Sun"""


@dataclass(frozen=True)
class WDelays:
    """Representation and mappings for the w-coordinate derived delays"""

    w_delays: u.Quantity
    """The w-derived delay. Shape is [baseline, time]"""
    b_map: dict[tuple[int, int], int]
    """The mapping between (ANTENNA1,ANTENNA2) to baseline index"""
    time_map: dict[float, int]
    """The mapping between time (MJDs from measurement set) to index"""


def get_sun_delay_for_ms(ms_path: Path) -> WDelays:
    # Generate the two sets of uvw coordinate objects
    baselines: Baselines = get_baselines_from_ms(ms_path=ms_path)
    hour_angles_phase: PositionHourAngles = make_hour_angles_for_ms(
        ms_path=ms_path,
        position=None,  # gets the position form phase direction
    )
    uvws_phase: UVWs = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles_phase)

    hour_angles_sun: PositionHourAngles = make_hour_angles_for_ms(
        ms_path=ms_path,
        position="sun",  # gets the position form phase direction
    )
    uvws_sun: UVWs = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles_sun)

    # Subtract the w-coordinates out. Since these uvws have
    # been computed towards different directions the difference
    # in w-coordinate is the delay distance
    w_diffs = uvws_sun.uvws[2] - uvws_phase.uvws[2]

    delay_object = (w_diffs / speed_of_light).decompose()

    return WDelays(
        w_delays=delay_object,
        b_map=baselines.b_map,
        time_map=hour_angles_phase.time_map,
    )


def dumb_tukey_tractor(
    tukey_tractor_options: TukeyTractorOptions,
) -> None:
    """Iterate row-wise over a specified measurement set and
    apply a tukey taper operation to the delay data. Iteration
    is performed based on a chunk soize, indicating the number
    of rows to read in at a time.

    Full description of options are outlined in `TukeyTaperOptions`.

    Args:
        tukey_tractor_options (TukeyTractorOptions): The settings to use during the taper, and measurement set to apply them to.
    """
    logger.info("jolly-roger")
    logger.info(f"Options: {tukey_tractor_options}")

    # acquire all the tables necessary to get unit information and data from
    open_ms_tables = get_open_ms_tables(
        ms_path=tukey_tractor_options.ms_path, read_only=False
    )

    if not tukey_tractor_options.dry_run:
        add_output_column(
            tab=open_ms_tables.main_table,
            output_column=tukey_tractor_options.output_column,
            data_column=tukey_tractor_options.data_column,
            overwrite=tukey_tractor_options.overwrite,
        )

    # Generate the delay for all baselines and time steps
    w_delays: WDelays | None = None
    if tukey_tractor_options.apply_towards_sun:
        logger.info("Pre-calculating delays towards the Sun")
        w_delays = get_sun_delay_for_ms(ms_path=tukey_tractor_options.ms_path)
        assert len(w_delays.w_delays.shape) == 2

    with tqdm(total=len(open_ms_tables.main_table)) as pbar:
        for data_chunk in get_data_chunks(
            open_ms_tables=open_ms_tables,
            chunk_size=tukey_tractor_options.chunk_size,
            data_column=tukey_tractor_options.data_column,
        ):
            taper_data_chunk: DataChunk = _tukey_tractor(
                data_chunk=data_chunk,
                tukey_tractor_options=tukey_tractor_options,
                w_delays=w_delays,
            )

            pbar.update(len(taper_data_chunk.masked_data))

            if tukey_tractor_options.dry_run:
                # Do not apply the data
                continue

            # only put if not a dry run
            open_ms_tables.main_table.putcol(
                columnname=tukey_tractor_options.output_column,
                value=taper_data_chunk.masked_data,
                startrow=taper_data_chunk.row_start,
                nrow=taper_data_chunk.chunk_size,
            )

    if tukey_tractor_options.make_plots and not tukey_tractor_options.dry_run:
        plot_paths: list[Path] = make_plot_results(
            open_ms_tables=open_ms_tables,
            data_column=tukey_tractor_options.data_column,
            output_column=tukey_tractor_options.output_column,
        )

        logger.info(f"Made {len(plot_paths)} output plots")


@dataclass
class TractorOptions:
    """Options for the Jolly Roger Tractor."""

    ms_path: Path
    """Path to the measurement set to process."""
    object: str = "sun"
    """Target position to use for the tractor. Defaults to 'sun'."""
    dry_run: bool = False
    """If set, the tractor will not write any output, but will log what it would do."""
    data_column: str = "DATA"
    """The data column to use for the tractor. Defaults to 'DATA'."""
    output_column: str = "CORRECTED_DATA"
    """The output column to write the tractor results to. Defaults to 'CORRECTED_DATA'."""
    make_plots: bool = False
    """If set, the tractor will make plots of the results."""


# def tractor_baseline(
#     ant_1: int,
#     ant_2: int,
#     baselines: Baselines,
#     uvws_object_baseline: u.Quantity,
#     options: TractorOptions,
# ) -> None:
#     open_ms_tables = get_open_ms_tables()
#     baseline_data = get_baseline_data(
#         baselines=baselines,
#         ant_1=ant_1,
#         ant_2=ant_2,
#         data_column=options.data_column,
#     )

#     delay_time = data_to_delay_time(data=baseline_data)

#     _, _, w_coords_object = uvws_object_baseline
#     _, _, w_coords_phase_center = baseline_data.uvws_phase_center

#     # Subtract
#     delay_object = (
#         (w_coords_object - w_coords_phase_center) / speed_of_light
#     ).decompose()
#     delay_phase_center = (
#         (w_coords_phase_center - w_coords_phase_center) / speed_of_light
#     ).decompose()

#     if options.make_plots:
#         output_dir = options.ms_path.parent / "plots"
#         output_dir.mkdir(exist_ok=True, parents=True)
#         plot_baseline_data(
#             baseline_data=baseline_data,
#             output_dir=output_dir,
#             suffix="_original",
#         )
#         plot_tractor_baseline(
#             baseline_data=baseline_data,
#             delay_time=delay_time,
#             delay_object=delay_object,
#             delay_phase_center=delay_phase_center,
#             output_dir=output_dir,
#         )
#         logger.info(f"Plots saved to {output_dir}")


def get_parser() -> ArgumentParser:
    """Create the CLI argument parser

    Returns:
        ArgumentParser: Constructed argument parser
    """
    parser = ArgumentParser(description="Run the Jolly Roger Tractor")
    subparsers = parser.add_subparsers(dest="mode")

    tukey_parser = subparsers.add_parser(
        name="tukey", help="Perform a dumb Tukey taper across delay-time data"
    )
    tukey_parser.add_argument(
        "ms_path",
        type=Path,
        help="The measurement set to process with the Tukey tractor",
    )
    tukey_parser.add_argument(
        "--outer-width",
        type=float,
        default=np.pi / 4,
        help="The outer width of the Tukey taper in radians",
    )
    tukey_parser.add_argument(
        "--tukey-width",
        type=float,
        default=np.pi / 8,
        help="The Tukey width of the Tukey taper in radians",
    )
    tukey_parser.add_argument(
        "--data-column",
        type=str,
        default="DATA",
        help="The data column to use for the Tukey tractor",
    )
    tukey_parser.add_argument(
        "--output-column",
        type=str,
        default="CORRECTED_DATA",
        help="The output column to write the Tukey tractor results to",
    )
    tukey_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, the Tukey tractor will not write any output, but will log what it would do",
    )
    tukey_parser.add_argument(
        "--make-plots",
        action="store_true",
        help="If set, the Tukey tractor will make plots of the results",
    )
    tukey_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, the Tukey tractor will overwrite the output column if it already exists",
    )
    tukey_parser.add_argument(
        "--chunk-size",
        type=int,
        default=10009,
        help="The number of rows to process in one chunk. Larger numbers require more memory but fewer interactions with I/O.",
    )
    tukey_parser.add_argument(
        "--apply-towards-sun",
        action="store_true",
        help="Whether the tukey taper is applied towards the sun",
    )

    return parser


def cli() -> None:
    """Command line interface for the Jolly Roger Tractor."""
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == "tukey":
        tukey_tractor_options = TukeyTractorOptions(
            ms_path=args.ms_path,
            outer_width=args.outer_width,
            tukey_width=args.tukey_width,
            data_column=args.data_column,
            output_column=args.output_column,
            dry_run=args.dry_run,
            make_plots=args.make_plots,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
            apply_towards_sun=args.apply_towards_sun,
        )
        dumb_tukey_tractor(tukey_tractor_options=tukey_tractor_options)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
