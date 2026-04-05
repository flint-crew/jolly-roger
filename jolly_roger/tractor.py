from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from pathlib import Path
from time import time
from typing import cast

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    SkyCoord,
)
from astropy.time import Time
from capn_crunch import BaseOptions, add_options_to_parser, create_options_from_parser
from casacore.tables import makecoldesc, table, taql
from numpy.typing import NDArray
from tqdm.auto import tqdm

from jolly_roger.baselines import (
    BaselineData,
    OpenMSTables,
    get_baseline_data,
    get_open_ms_tables,
)
from jolly_roger.delays import DelayTime, data_to_delay_time, delay_time_to_data
from jolly_roger.logging import logger
from jolly_roger.plots import plot_baseline_comparison_data
from jolly_roger.utils import log_dataclass_attributes, log_jolly_roger_version
from jolly_roger.uvws import WDelays, get_object_delay_for_ms
from jolly_roger.wrap import calculate_nyquist_zone, symmetric_domain_wrap


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


def get_multiple_data_chunks(
    open_ms_tables: OpenMSTables,
    chunk_size: int,
    data_column: str,
    number_of_chunks: int = 1,
) -> Generator[tuple[DataChunk, ...], None, None]:
    """
    Wrapper around ``get_data_chunks`` to yield a list of
    data chunks.

    Each data chunk is a collection of rows with appropriate units
    attached to the quantities. These quantities are not
    the same data encoded in the measurement set, e.g.
    masked array has been formed, astropy units have
    been attached.

    Args:
        open_ms_tables (OpenMSTables): References to open tables from the measurement set
        chunk_size (int): The number of rows to return at a time
        data_column (str): The data column that would be modified
        number_of_chunks (int, optional): The number of chunks to return on each yield. Defaults to 1.

    Yields:
        Generator[tuple[DataChunk, ...], None, None]: Representation of the current chunk of rows
    """
    # To hold the set of data chunks
    base_chunks: list[DataChunk] = []

    # We are using the existing generator. Termination should be
    # straightforward.
    for data_chunk in get_data_chunks(
        open_ms_tables=open_ms_tables, chunk_size=chunk_size, data_column=data_column
    ):
        base_chunks.append(data_chunk)

        # Once we hit the numbner of requested chunks return them, then empty
        if len(base_chunks) == number_of_chunks:
            yield tuple(base_chunks)
            base_chunks = []

    # Throw out any left overs
    if len(base_chunks) > 0:
        yield tuple(base_chunks)


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
    target: str | None = None,
    w_delays: WDelays | list[WDelays] | None = None,
    reverse_baselines: bool = False,
    outer_width_ns: float | None = None,
    max_baselines: int = 10,
) -> list[Path]:
    """Create plots useful for diagnostics

    Args:
        open_ms_tables (OpenMSTables): Collection of open MS tables describing data to be modified
        data_column (str): The 'before' data
        output_column (str): The output 'after' data
        target (str): Object nulling was directed towards
        w_delays (WDelays | None, optional): Description of a track through delay space. If ``None`` some plotting will be skipped. Defaults to None.
        reverse_baselines (bool, optional): Needed in some circumstances should antenna ordering in MS be different. Defaults to False.
        outer_width_ns (float | None, optional): Size, in nanoseconds, of the tukey taper. Defaults to None.
        max_baselines (int, optional): The maximum number of baseline plots to create. Defaults to 10.

    Returns:
        list[Path]: Collection of paths to use
    """
    assert max_baselines > 0, f"{max_baselines=}, but should be at least 0"

    output_paths = []
    output_dir = open_ms_tables.ms_path.parent / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    n_ant = len(np.unique(open_ms_tables.main_table.getcol("ANTENNA1")))
    b_idx = np.array(list(combinations(range(n_ant), 2)))

    logger.info(f"MS contains {n_ant} antennas ({len(b_idx)} baselines)")

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
        name_components = [
            ms_name,
            "baseline_data",
            f"{before_baseline_data.ant_1}",
            f"{before_baseline_data.ant_2}",
        ]
        if target:
            name_components.append(target)
        elif target is None and w_delays is not None:
            name_components.append(
                w_delays.object_name if isinstance(w_delays, WDelays) else "multi"
            )
        else:
            name_components.append("none")

        name_components.append("comparison.png")
        output_path = output_dir / f"{'_'.join(name_components)}"

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
            for ant_1, ant_2 in zip(data_chunk.ant_1, data_chunk.ant_2, strict=False)
        ]
    )

    time_idx = np.array(
        [w_delays.time_map[time * u.s] for time in data_chunk.time_mjds]
    )

    return baseline_idx, time_idx


@dataclass
class TaperResult:
    """Simple container to help ensure consistent and trackable behaviour
    across levels
    """

    attached_payload: bool = False
    """Indicates whether anything to do."""
    taper: NDArray[np.floating] | None = None
    """The taper to apply. If None nothing to do."""
    update_flags: bool = False
    """Indicates whether flags need to be updated"""
    flags: NDArray[bool] | None = None
    """The fupdated flags"""
    delay_time: DelayTime | None = None
    """The delay_time taper was constructede against"""


def _tukey_tractor(
    data_chunk: DataChunk,
    tukey_tractor_options: TukeyTractorOptions,
    w_delays: WDelays,
    delay_time: DelayTime | None = None,
) -> TaperResult:
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
        delay_time (DelayTime | None, optional): Optional pre-computed DelayTime object.

    Returns:
        TaperResult: Scaled complex visibilities, corresponding flags, and delays.
    """

    baseline_idx, time_idx = _get_baseline_time_indicies(
        w_delays=w_delays, data_chunk=data_chunk
    )

    # Delay with the elevation of the target object
    elevation_mask = w_delays.elevation < (
        tukey_tractor_options.elevation_cut_deg * u.deg
    )
    # Bail out early if there is nothing that can be done with this source
    if np.all(elevation_mask[time_idx]):
        return TaperResult(attached_payload=False)

    if delay_time is None:
        delay_time = data_to_delay_time(data=data_chunk)

    # Set up the offsets. By default we will be tapering around the field,
    # but should w_delays be specified these will be modified to direct
    # towards the nominated object in the if below
    tukey_x_offset: u.Quantity = np.zeros_like(delay_time.delay)

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
    # taper shape is [chunk_size, no_channels, no_pols]

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

    taper[elevation_mask[time_idx], :, :] = 1.0

    # Compute flags to ignore the objects delay crossing 0, Do
    # This by computing the taper towards the field and
    # see if there are any components of the two sets of tapers
    # that are not 1 (where 1 is 'no change').
    field_taper = tukey_taper(
        x=delay_time.delay.to("s").value,
        outer_width=tukey_tractor_options.outer_width_ns * 1e-9,
        tukey_width=tukey_tractor_options.tukey_width_ns * 1e-9,
        tukey_x_offset=None,
    )
    # field_taper.shape is [no_channels, ]
    # We need to account for no broadcasting when offset is None
    # as the returned shape is different
    field_taper = field_taper[None, :, None]
    field_taper = 1.0 - field_taper
    intersecting_taper = np.any(
        np.reshape((taper != 1) & (field_taper != 1), (taper.shape[0], -1)), axis=1
    )

    # Here we consider what to do if want to compare brightness of the object in delay
    # space is less than that of the field. If the object is not detected we ought to
    # set the tape to 1 so the data are not modified
    if tukey_tractor_options.compare_to_field:
        # The delay spectrum are complex quantities, and we need to compare
        # the flux
        # Make a stokes I type spectrum
        # logger.info(f"{delay_time.delay_time.shape=}")
        abs_delay_time = np.abs(np.sum(delay_time.delay_time[..., [0, -1]], axis=-1))
        # output of the field taper is constant over rows, so
        # some broadcasting is needed to handle the array shapes
        _field_taper = np.squeeze(field_taper)
        field_stats = np.max(abs_delay_time * (1.0 - _field_taper)[None, :], axis=1)
        object_stats = np.max(abs_delay_time * (1.0 - taper[..., 0]), axis=1)

        # Depending on size of chunk this could be expensive
        logger.debug("np.sum(_field_taper)=%f", np.sum(_field_taper))
        logger.debug("np.sum(taper[0, :, 0])=%f", np.sum(taper[0, :, 0]))

        flux_mask = object_stats < tukey_tractor_options.compare_to_field * field_stats

        # For any element where there is not enough flux set the taper so
        # it does not modify the data
        taper[flux_mask, :] = 1.0

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

    return TaperResult(
        attached_payload=True,
        taper=taper,
        update_flags=np.any(flags_to_return),
        flags=flags_to_return,
        delay_time=delay_time,
    )


def _apply_taper(
    data_chunk: DataChunk,
    delay_time: DelayTime,
    taper: NDArray[np.float64],
) -> DataChunk:
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

    return tapered_data


@dataclass
class TaperedChunkResult:
    """ "Simple container for the application of tapered data and associated
    meta-data to write back to the MS
    """

    chunk_size: int
    """The size of the chunk this result represents"""
    nothing_to_do: bool = False
    """Simple flag indicating this chunk contains nothing that needs updating so can be skipped"""
    update_data: bool = False
    """Indicates whether the data chunk needs to be written back"""
    data_chunk: DataChunk | None = None
    """The data to write back"""
    update_flags: bool = False
    """"Indicates whether the flags need to be written back to MS"""
    flags: NDArray[bool] | None = None
    """The flags to write back"""


def _tukey_multi_tractor(
    data_chunk: DataChunk,
    tukey_tractor_options: TukeyTractorOptions,
    w_delays_list: list[WDelays],
) -> TaperedChunkResult:
    chunk_size = data_chunk.chunk_size

    # Get the results for each object. Note reusing the delay_time object
    # to avoid unnecessary recomputes. Also why the for loop and not list comphrension
    delay_time: DelayTime | None = None
    taper_results = []
    for w_delays in w_delays_list:
        taper_result = _tukey_tractor(
            data_chunk=data_chunk,
            tukey_tractor_options=tukey_tractor_options,
            w_delays=w_delays,
            delay_time=delay_time,
        )
        delay_time = taper_result.delay_time
        taper_results.append(taper_result)

    # Handle all the cases
    if all(not taper_result.attached_payload for taper_result in taper_results):
        return TaperedChunkResult(chunk_size=chunk_size, nothing_to_do=True)

    # Throw away objects that are unnecessary in subsequent stages
    taper_results = [
        taper_result for taper_result in taper_results if taper_result.attached_payload
    ]

    combined_taper = np.min(
        [
            taper_result.taper
            for taper_result in taper_results
            if taper_result.taper is not None
        ],
        axis=0,
    )

    update_flag_list = [
        taper_result.flags
        for taper_result in taper_results
        if taper_result.update_flags
    ]
    update_flags = len(update_flag_list) > 0

    combined_flags = (
        np.sum(update_flag_list, axis=0).astype(bool) if update_flags else None
    )

    tapered_data = _apply_taper(
        data_chunk=data_chunk,
        delay_time=cast(DelayTime, delay_time),
        taper=combined_taper,
    )

    return TaperedChunkResult(
        chunk_size=chunk_size,
        nothing_to_do=False,
        update_data=True,
        data_chunk=tapered_data,
        update_flags=update_flags,
        flags=combined_flags,
    )


class TukeyTractorOptions(BaseOptions):
    """Options to describe the tukey taper to apply"""

    target_objects: tuple[str, ...] = ("sun",)
    """The target object to apply the delay towards."""
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
    number_of_plots: int = 10
    """The number of output plots to make. Defaults to 10."""
    overwrite: bool = False
    """If the output column exists it will be overwritten"""
    chunk_size: int = 1000
    """Size of the row-wise chunking iterator"""
    elevation_cut_deg: float = -1.0
    """The elevation cut-off for the target object in degrees. Defaults to -1 degrees."""
    ignore_nyquist_zone: int = 2
    """Do not apply the tukey taper if object is beyond this Nyquist zone"""
    reverse_baselines: bool = False
    """Reverse baseline ordering"""
    flip_uvw_sign: bool = False
    """Flip the sign of UVWs (required for LOFAR)"""
    max_workers: int = 1
    """The number of compute processes to establish. Each process gets chunk_size of rows. If max_worker==1 all work is performed in main thread."""
    compare_to_field: float | None = None
    """Compare the source brightness in delay space to the field. If the source is fainter than the field multiplied by this factor, do not taper. Defaults to None."""


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
    ms_path: Path,
    tukey_tractor_options: TukeyTractorOptions,
) -> TukeyTractorResults:
    """Iterate row-wise over a specified measurement set and
    apply a tukey taper operation to the delay data. Iteration
    is performed based on a chunk size, indicating the number
    of rows to read in at a time.

    Full description of options are outlined in `TukeyTaperOptions`.

    Args:
        ms_path (Path): The MS to be modified.
        tukey_tractor_options (TukeyTractorOptions): The settings to use during the taper, and measurement set to apply them to.

    Returns:
        TukeyTractorResults: Representative information of the tapering process
    """
    log_jolly_roger_version()
    log_dataclass_attributes(
        to_log=tukey_tractor_options, class_name="TukeyTaperOptions"
    )

    # acquire all the tables necessary to get unit information and data from
    open_ms_tables = get_open_ms_tables(ms_path=ms_path, read_only=False)

    if not tukey_tractor_options.dry_run:
        add_output_column(
            tab=open_ms_tables.main_table,
            output_column=tukey_tractor_options.output_column,
            data_column=tukey_tractor_options.data_column,
            overwrite=tukey_tractor_options.overwrite,
            copy_column_data=tukey_tractor_options.copy_column_data,
        )

    # Generate the delay for all baselines and time steps
    w_delays_list = get_object_delay_for_ms(
        ms_path=ms_path,
        object_name=tukey_tractor_options.target_objects,
        reverse_baselines=tukey_tractor_options.reverse_baselines,
        flip_uvw_sign=tukey_tractor_options.flip_uvw_sign,
    )
    assert all(len(w_delays.w_delays.shape) == 2 for w_delays in w_delays_list), (
        "Sanity check failed, incorrect dimensionality returned"
    )

    if not tukey_tractor_options.dry_run:
        pool: None | ThreadPoolExecutor = None
        if tukey_tractor_options.max_workers > 1:
            logger.info(
                f"Starting {tukey_tractor_options.max_workers} computer workers. Be mindful of {tukey_tractor_options.chunk_size=}"
            )
            pool = ThreadPoolExecutor(max_workers=tukey_tractor_options.max_workers)

        # This could be offloaded to some other function if we end up
        # havig multiple modes
        partial_compute_func = partial(
            _tukey_multi_tractor,
            tukey_tractor_options=tukey_tractor_options,
            w_delays_list=w_delays_list,
        )

        start = time()
        total_tukey_time_s = 0.0
        with tqdm(total=len(open_ms_tables.main_table), desc="Rows") as pbar:
            for data_chunk_tuple in get_multiple_data_chunks(
                open_ms_tables=open_ms_tables,
                chunk_size=tukey_tractor_options.chunk_size,
                data_column=tukey_tractor_options.data_column,
                number_of_chunks=tukey_tractor_options.max_workers,
            ):
                start_tukey = time()
                taper_data_and_flags: list[TaperedChunkResult] = []
                if len(data_chunk_tuple) == 1:
                    taper_results = partial_compute_func(
                        data_chunk=data_chunk_tuple[0],
                    )
                    taper_data_and_flags = [taper_results]
                else:
                    assert pool is not None, f"{pool=}, and should not be None"
                    # This creates an iterator that will return objects as they are completed,
                    # but breaks timing stats. Not a clear increase of throughput in (brief) testing
                    # taper_data_and_flags = pool.map(partial_compute_func, data_chunk_tuple)

                    # The list collects all results in order.
                    taper_data_and_flags = list(
                        pool.map(partial_compute_func, data_chunk_tuple)
                    )

                end_tukey = time()
                total_tukey_time_s += end_tukey - start_tukey

                # Iterate over the result set in a serial manner. Note that depending on
                # how the .map is called in max_workers>1 case this could be a generator
                # for taper_data_chunk, flags_to_apply in taper_data_and_flags:
                taper_chunk_result: TaperedChunkResult
                for taper_chunk_result in taper_data_and_flags:
                    pbar.update(taper_chunk_result.chunk_size)

                    if taper_chunk_result.nothing_to_do:
                        logger.debug("No attached data payload, skipping")
                        continue

                    data_chunk = taper_chunk_result.data_chunk
                    assert data_chunk is not None, (
                        f"{data_chunk=}, which should not happen"
                    )

                    # Only update here is we pass the dry run check above
                    if taper_chunk_result.update_data:
                        open_ms_tables.main_table.putcol(
                            columnname=tukey_tractor_options.output_column,
                            value=data_chunk.masked_data,
                            startrow=data_chunk.row_start,
                            nrow=data_chunk.chunk_size,
                        )
                    else:
                        logger.debug("No update of data required, skipping")
                    if taper_chunk_result.update_flags:
                        open_ms_tables.main_table.putcol(
                            columnname="FLAG",
                            value=taper_chunk_result.flags,
                            startrow=data_chunk.row_start,
                            nrow=data_chunk.chunk_size,
                        )
                    else:
                        logger.debug(
                            "No updating of flags required, skipping for chunk"
                        )
        stop = time()
        runtime_s = stop - start
        assert data_chunk is not None, "data_chunk is not formed correctly"
        logger.info(
            f"Tapered {len(tukey_tractor_options.target_objects)} targets over {len(open_ms_tables.main_table)} rows by {len(data_chunk.freq_chan)} chans in {runtime_s:0.2f}s"
        )

        logger.info(
            f"Nulling time: {total_tukey_time_s:0.2f}s, reading/writing: {runtime_s - total_tukey_time_s:0.2f}s"
        )
        if tukey_tractor_options.max_workers > 1:
            logger.info(
                f"Used {tukey_tractor_options.max_workers} workers with a {tukey_tractor_options.chunk_size} chunk size"
            )

        if isinstance(pool, ThreadPoolExecutor):
            logger.info("Closing thread pool...")
            pool.shutdown()

    plot_paths: list[Path] | None
    if tukey_tractor_options.make_plots:
        plot_paths = make_plot_results(
            open_ms_tables=open_ms_tables,
            data_column=tukey_tractor_options.data_column,
            output_column=tukey_tractor_options.output_column,
            w_delays=w_delays_list,
            reverse_baselines=tukey_tractor_options.reverse_baselines,
            outer_width_ns=tukey_tractor_options.outer_width_ns,
            max_baselines=tukey_tractor_options.number_of_plots,
        )

        logger.info(f"Made {len(plot_paths)} output plots")
    else:
        plot_paths = None

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
        name="tukey",
        help="Perform a simple Tukey taper across delay-time data",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    tukey_parser.add_argument(
        "ms_path",
        type=Path,
        help="The measurement set to process with the Tukey tractor",
    )
    tukey_parser = add_options_to_parser(
        parser=tukey_parser,
        options_class=TukeyTractorOptions,
    )

    return parser


def cli() -> None:
    """Command line interface for the Jolly Roger Tractor."""
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == "tukey":
        tukey_tractor_options = create_options_from_parser(
            parser_namespace=args,
            options_class=TukeyTractorOptions,
        )

        tukey_tractor(ms_path=args.ms_path, tukey_tractor_options=tukey_tractor_options)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
