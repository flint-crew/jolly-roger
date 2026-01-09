from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Generator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    SkyCoord,
)
from astropy.time import Time
from casacore.tables import makecoldesc, table, taql
from numpy.typing import NDArray
from tqdm.auto import tqdm

from jolly_roger.delays import data_to_delay_time, delay_time_to_data
from jolly_roger.logging import logger
from jolly_roger.plots import plot_baseline_comparison_data
from jolly_roger.utils import log_dataclass_attributes, log_jolly_roger_version
from jolly_roger.uvws import WDelays, get_object_delay_for_ms
from jolly_roger.wrap import calculate_nyquist_zone, symmetric_domain_wrap


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


def tukey_taper(
    x: np.typing.NDArray[np.floating],
    outer_width: float,
    tukey_width: float,
    tukey_x_offset: NDArray[np.floating] | None = None,
) -> np.ndarray:
    """Describes a tukey window function spanning a -x.min() to x.max() range. In the base case
    the tukey window is centred on 0.0. The ``outer_width`` defines where the window is
    0.0. They `tukey_width` defines the width of the region where the function transitions
    from 1.0 to 0.0.


    This is to say that:

    >> x > |outer_width| = 0
    >> x < (outer_width - tukey_width) = 1

    Between these two bounds the window follows a `1 - cos` type shape.

    Args:
        x (np.typing.NDArray[np.floating]): The intervals to evaluate over. Internally these are concerted to the +/- pi domain
        outer_width (float, optional): The +/- boundary beyond which is 0.0.
        tukey_width (float, optional): Describes the width that the transition from 1.0 to 0.0 occurs.
        tukey_x_offset (NDArray[np.floating] | None, optional): Sets a new zero point (center of window). Defaults to None.
        notch (bool, optional): Will the taper be used for a notch filter? Defaults to True.

    Returns:
        np.ndarray: The tukey window function
    """
    # Copy to avoid side-effects
    x_local = x.copy()

    if (outer_width - tukey_width) < 0.0:
        # If this is true than the two 'transition' regions between 1 and 0 overlap.
        # This should not happen, so we simply will make it so no '1' region. In this extreme
        # the window is just a 1 - cos function
        logger.warning(
            f"{outer_width=} and {tukey_width=}, which create overlapping bounds. Setting tukey_width={outer_width}"
        )
        tukey_width = outer_width

    if tukey_x_offset is not None:
        # Save the original maximum of the input domain before any shifting ...
        original_x_local_maximum = np.max(x_local)
        x_local = x_local[:, None] - tukey_x_offset[None, :]

        # ... so that the original maximum is used in the unwrapping
        x_local = symmetric_domain_wrap(
            values=x_local, upper_limit=original_x_local_maximum
        )

    taper = np.ones_like(x_local)
    # Fully zero region
    taper[np.abs(x_local) > outer_width] = 0

    # Transition regions
    left_idx = (-outer_width <= x_local) & (x_local <= -outer_width + tukey_width)
    right_idx = (outer_width - tukey_width <= x_local) & (x_local <= outer_width)

    taper[left_idx] = (
        1 - np.cos(np.pi * (x_local[left_idx] + outer_width) / tukey_width)
    ) / 2

    taper[right_idx] = (
        1 - np.cos(np.pi * (outer_width - x_local[right_idx]) / tukey_width)
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

    while lower_row < table_length:
        data = ms_table.getcol(data_column, startrow=lower_row, nrow=chunk_size)
        flags = ms_table.getcol("FLAG", startrow=lower_row, nrow=chunk_size)
        uvws = ms_table.getcol("UVW", startrow=lower_row, nrow=chunk_size)
        time_centroid = ms_table.getcol(
            "TIME_CENTROID", startrow=lower_row, nrow=chunk_size
        )
        ant_1 = ms_table.getcol("ANTENNA1", startrow=lower_row, nrow=chunk_size)
        ant_2 = ms_table.getcol("ANTENNA2", startrow=lower_row, nrow=chunk_size)

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
    else:
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
    open_ms_tables: OpenMSTables,
    data_column: str,
    output_column: str,
    target: str,
    w_delays: WDelays | None = None,
    reverse_baselines: bool = False,
    outer_width_ns: float | None = None,
) -> list[Path]:
    output_paths = []
    output_dir = open_ms_tables.ms_path.parent / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    n_ant = len(np.unique(open_ms_tables.main_table.getcol("ANTENNA1")))
    b_idx = np.array(list(combinations(range(n_ant), 2)))

    logger.info(f"MS contains {n_ant} antennas ({len(b_idx)} baselines)")

    max_baselines = 10
    b_idx = b_idx[:max_baselines]
    logger.info(f"Plotting {len(b_idx)} baselines")

    if reverse_baselines:
        b_idx = b_idx[:, ::-1]

    for baseline, (ant_1, ant_2) in enumerate(b_idx):
        logger.info(f"Plotting baseline={baseline + 1}")

        before_baseline_data = get_baseline_data(
            open_ms_tables=open_ms_tables,
            ant_1=ant_1,
            ant_2=ant_2,
            data_column=data_column,
        )
        after_baseline_data = get_baseline_data(
            open_ms_tables=open_ms_tables,
            ant_1=ant_1,
            ant_2=ant_2,
            data_column=output_column,
        )
        before_delays = data_to_delay_time(data=before_baseline_data)
        after_delays = data_to_delay_time(data=after_baseline_data)

        ms_name = open_ms_tables.ms_path.name
        output_path = (
            output_dir
            / f"{ms_name}_baseline_data_{before_baseline_data.ant_1}_{before_baseline_data.ant_2}_{target}_comparison.png"
        )

        logger.info("Creating figure")
        # TODO: the baseline data and delay times could be put into a single
        # structure to pass around easier.
        plot_path = plot_baseline_comparison_data(
            before_baseline_data=before_baseline_data,
            after_baseline_data=after_baseline_data,
            before_delays=before_delays,
            after_delays=after_delays,
            output_path=output_path,
            w_delays=w_delays,
            outer_width_ns=outer_width_ns,
        )
        logger.info(f"Have written {output_path=}")
        output_paths.append(plot_path)

    return output_paths


def _get_baseline_time_indicies(
    w_delays: WDelays, data_chunk: DataChunk
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Extract the mappings into the data array"""

    # When computing uvws we have ignored auto-correlations!
    # TODO: Either extend the uvw calculations to include auto-correlations
    # or ignore them during iterations. Certainly the former is the better
    # approach.

    # Again, note the auto-correlations are ignored!!! Here be pirates mate
    baseline_idx = np.array(
        [
            w_delays.b_map[(int(ant_1), int(ant_2))] if ant_1 != ant_2 else 0
            for ant_1, ant_2 in zip(  # type: ignore[call-overload]
                data_chunk.ant_1, data_chunk.ant_2, strict=False
            )
        ]
    )

    time_idx = np.array(
        [w_delays.time_map[time * u.s] for time in data_chunk.time_mjds]
    )

    return baseline_idx, time_idx


def _tukey_tractor(
    data_chunk: DataChunk,
    tukey_tractor_options: TukeyTractorOptions,
    w_delays: WDelays,
) -> tuple[DataChunk, NDArray[np.bool] | None]:
    """Compute a tukey taper for a dataset and then apply it
    to the dataset. Here the data corresponds to a (chan, time, pol)
    array. Data is not necessarily a single baseline.

    The provided ``w_delays`` describes the object that nulling will be
    centred towards. This quantity may be derived in a number of ways, but
    in ``jolly_roger`` it is based on the difference of the w-coordinated
    towards these two directions. It should have a shape of [baselines, time]

    Args:
        data_chunk (DataChunk): The representation of the data with attached units
        tukey_tractor_options (TukeyTractorOptions): Options for the tukey taper
        w_delays (WDelays): The w-derived delays to apply.

    Returns:
        tuple[DataChunk,NDArray[np.bool] | None]: Scaled complex visibilities and corresponding flags. If flags do not need to be updated ``None`` is returned.
    """

    delay_time = data_to_delay_time(data=data_chunk)
    flags_to_return: None | NDArray[np.bool] = None

    # Set up the offsets. By default we will be tapering around the field,
    # but should w_delays be specified these will be modified to direct
    # towards the nominated object in the if below
    tukey_x_offset: u.Quantity = np.zeros_like(delay_time.delay)

    baseline_idx, time_idx = _get_baseline_time_indicies(
        w_delays=w_delays, data_chunk=data_chunk
    )
    original_tukey_x_offset = w_delays.w_delays[baseline_idx, time_idx]

    # Make a copy for later use post wrapping
    tukey_x_offset = original_tukey_x_offset.copy()

    # need to scale the x offset to the -pi to pi (radians) wrap
    # keeping units in seconds though
    # The delay should be symmetric
    tukey_x_offset_sec = symmetric_domain_wrap(
        values=tukey_x_offset.to("s").value,
        upper_limit=np.max(delay_time.delay).to("s").value,
    )

    # Make taper with all units in seconds
    taper = tukey_taper(
        x=delay_time.delay.to("s").value,
        outer_width=tukey_tractor_options.outer_width_ns * 1e-9,
        tukey_width=tukey_tractor_options.tukey_width_ns * 1e-9,
        tukey_x_offset=tukey_x_offset_sec,
    )

    # TODO: This pirate reckons that merging the masks together
    # into a single mask throughout may make things easier to
    # manage and visualise.

    # The use of the `tukey_x_offset` changes the
    # shape of the output array. The internals of that
    # function returns a different shape via the broadcasting
    taper = np.swapaxes(taper[:, :, None], 0, 1)

    # Since we want to dampen the target object we invert the taper.
    # By default the taper dampers outside the inner region.
    taper = 1.0 - taper

    # apply the flags to ignore the tapering if the object is larger
    # than one wrap away
    # Calculate the offset account of nyquist sampling
    no_wraps_for_offset = calculate_nyquist_zone(
        values=original_tukey_x_offset.value,
        upper_limit=np.max(delay_time.delay).value,
    )
    ignore_wrapping_for = (
        no_wraps_for_offset > tukey_tractor_options.ignore_nyquist_zone
    )
    taper[ignore_wrapping_for, :, :] = 1.0

    # Delay with the elevation of the target object
    elevation_mask = w_delays.elevation < tukey_tractor_options.elevation_cut
    taper[elevation_mask[time_idx], :, :] = 1.0

    # Compute flags to ignore the objects delay crossing 0, Do
    # This by computing the taper towards the field and
    # see if there are any components of the two sets of tapers
    # that are not 1 (where 1 is 'no change').
    field_taper = tukey_taper(
        x=delay_time.delay.to("s").value,
        outer_width=tukey_tractor_options.outer_width_ns * 1e-9 / 4,
        tukey_width=tukey_tractor_options.tukey_width_ns * 1e-9 / 4,
        tukey_x_offset=None,
    )
    # We need to account for no broadcasting when offset is None
    # as the returned shape is different
    field_taper = field_taper[None, :, None]
    field_taper = 1.0 - field_taper
    intersecting_taper = np.any(
        np.reshape((taper != 1) & (field_taper != 1), (taper.shape[0], -1)), axis=1
    )
    # # Should the data need to be modified in conjunction with the flags
    # taper[
    #     intersecting_taper &
    #     ~elevation_mask[time_idx] &
    #     ~ignore_wrapping_for
    # ] = 0.0
    # Update flags
    flags_to_return = np.zeros_like(data_chunk.masked_data.mask)
    flags_to_return[intersecting_taper] = True
    flags_to_return = (
        ~np.isfinite(data_chunk.masked_data.filled(np.nan)) | flags_to_return
    )

    # Delay-time is a 3D array: (time, delay, pol)
    # Taper is 1D: (delay,)
    tapered_delay_time_data_real = delay_time.delay_time.real * taper
    tapered_delay_time_data_imag = delay_time.delay_time.imag * taper
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

    return tapered_data, flags_to_return


@dataclass
class TukeyTractorOptions:
    """Options to describe the tukey taper to apply"""

    ms_path: Path
    """Measurement set to be modified"""
    outer_width_ns: float = 10
    """The start of the tapering in nanoseconds"""
    tukey_width_ns: float = 10
    """The width of the tapered region in nanoseconds"""
    data_column: str = "DATA"
    """The visibility column to modify"""
    output_column: str = "CORRECTED_DATA"
    """The output column to be created with the modified data"""
    copy_column_data: bool = False
    """Copy the data from the data column to the output column before applying the taper"""
    dry_run: bool = False
    """Indicates whether the data will be written back to the measurement set"""
    make_plots: bool = False
    """Create a small set of diagnostic plots. This can be slow."""
    overwrite: bool = False
    """If the output column exists it will be overwritten"""
    chunk_size: int = 1000
    """Size of the row-wise chunking iterator"""
    target_object: str = "Sun"
    """The target object to apply the delay towards."""
    elevation_cut: u.Quantity = -1 * u.deg
    """The elevation cut-off for the target object. Defaults to 0 degrees."""
    ignore_nyquist_zone: int = 2
    """Do not apply the tukey taper if object is beyond this Nyquist zone"""
    reverse_baselines: bool = False
    """Reverse baseline ordering"""


@dataclass(frozen=True)
class TukeyTractorResults:
    """Simple return set of results from the tractoring process"""

    ms_path: Path
    """Path to the measurement set that was modified"""
    output_column: str
    """The name of the column that has the modified/tapered visibilities"""
    output_plots: list[Path] | None = None
    """The output plots that were created, if any"""


def tukey_tractor(
    tukey_tractor_options: TukeyTractorOptions,
) -> TukeyTractorResults:
    """Iterate row-wise over a specified measurement set and
    apply a tukey taper operation to the delay data. Iteration
    is performed based on a chunk soize, indicating the number
    of rows to read in at a time.

    Full description of options are outlined in `TukeyTaperOptions`.

    Args:
        tukey_tractor_options (TukeyTractorOptions): The settings to use during the taper, and measurement set to apply them to.

    Returns:
        TukeyTractorResults: Representative information of the tapering process
    """
    log_jolly_roger_version()
    log_dataclass_attributes(
        to_log=tukey_tractor_options, class_name="TukeyTaperOptions"
    )

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
            copy_column_data=tukey_tractor_options.copy_column_data,
        )

    # Generate the delay for all baselines and time steps
    logger.info(
        f"Pre-calculating delays towards the target: {tukey_tractor_options.target_object}"
    )
    w_delays = get_object_delay_for_ms(
        ms_path=tukey_tractor_options.ms_path,
        object_name=tukey_tractor_options.target_object,
        reverse_baselines=tukey_tractor_options.reverse_baselines,
    )
    assert len(w_delays.w_delays.shape) == 2

    if not tukey_tractor_options.dry_run:
        with tqdm(total=len(open_ms_tables.main_table)) as pbar:
            for data_chunk in get_data_chunks(
                open_ms_tables=open_ms_tables,
                chunk_size=tukey_tractor_options.chunk_size,
                data_column=tukey_tractor_options.data_column,
            ):
                taper_data_chunk, flags_to_apply = _tukey_tractor(
                    data_chunk=data_chunk,
                    tukey_tractor_options=tukey_tractor_options,
                    w_delays=w_delays,
                )

                pbar.update(len(taper_data_chunk.masked_data))

                # Only update here is we pass the dry run check above
                open_ms_tables.main_table.putcol(
                    columnname=tukey_tractor_options.output_column,
                    value=taper_data_chunk.masked_data,
                    startrow=taper_data_chunk.row_start,
                    nrow=taper_data_chunk.chunk_size,
                )
                if flags_to_apply is not None:
                    open_ms_tables.main_table.putcol(
                        columnname="FLAG",
                        value=flags_to_apply,
                        startrow=taper_data_chunk.row_start,
                        nrow=taper_data_chunk.chunk_size,
                    )

    plot_paths: list[Path] | None = None
    if tukey_tractor_options.make_plots:
        plot_paths = make_plot_results(
            open_ms_tables=open_ms_tables,
            data_column=tukey_tractor_options.data_column,
            output_column=tukey_tractor_options.output_column,
            target=tukey_tractor_options.target_object,
            w_delays=w_delays,
            reverse_baselines=tukey_tractor_options.reverse_baselines,
            outer_width_ns=tukey_tractor_options.outer_width_ns,
        )

        logger.info(f"Made {len(plot_paths)} output plots")

    return TukeyTractorResults(
        ms_path=open_ms_tables.ms_path,
        output_column=tukey_tractor_options.output_column,
        output_plots=plot_paths,
    )


def get_parser() -> ArgumentParser:
    """Create the CLI argument parser

    Returns:
        ArgumentParser: Constructed argument parser
    """
    parser = ArgumentParser(
        description="Run the Jolly Roger Tractor",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode")

    tukey_parser = subparsers.add_parser(
        name="tukey", help="Perform a simple Tukey taper across delay-time data"
    )
    tukey_parser.add_argument(
        "ms_path",
        type=Path,
        help="The measurement set to process with the Tukey tractor",
    )
    tukey_parser.add_argument(
        "--outer-width",
        type=float,
        default=10,
        help="The outer width of the Tukey taper in nanoseconds. If unset defaults to --tukey-width",
    )
    tukey_parser.add_argument(
        "--tukey-width",
        type=float,
        default=None,
        help="The Tukey width of the Tukey taper in nanoseconds",
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
        "--copy-column-data",
        action="store_true",
        help="If set, the Tukey tractor will copy the data from the data column to the output column before applying the taper",
    )
    tukey_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, the Tukey tractor will not write any output, but will log what it would do",
    )
    tukey_parser.add_argument(
        "--make-plots",
        action="store_true",
        help="If set, the Tukey tractor will make plots of the results. This can be slow.",
    )
    tukey_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, the Tukey tractor will overwrite the output column if it already exists",
    )
    tukey_parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="The number of rows to process in one chunk. Larger numbers require more memory but fewer interactions with I/O.",
    )
    tukey_parser.add_argument(
        "--target-object",
        type=str,
        default="Sun",
        help="The target object to apply the delay towards. Defaults to 'Sun'.",
    )
    tukey_parser.add_argument(
        "--ignore-nyquist-zone",
        type=int,
        default=2,
        help="Do not apply the taper if the objects delays beyond this Nyquist zone",
    )
    tukey_parser.add_argument(
        "--reverse-baselines",
        action="store_true",
        help="Reverse baseline ordering",
    )

    return parser


def cli() -> None:
    """Command line interface for the Jolly Roger Tractor."""
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == "tukey":
        tukey_tractor_options = TukeyTractorOptions(
            ms_path=args.ms_path,
            outer_width_ns=args.outer_width,
            tukey_width_ns=args.tukey_width
            if args.tukey_width is not None
            else args.outer_width,
            data_column=args.data_column,
            output_column=args.output_column,
            copy_column_data=args.copy_column_data,
            dry_run=args.dry_run,
            make_plots=args.make_plots,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
            target_object=args.target_object,
            ignore_nyquist_zone=args.ignore_nyquist_zone,
            reverse_baselines=args.reverse_baselines,
        )

        tukey_tractor(tukey_tractor_options=tukey_tractor_options)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
