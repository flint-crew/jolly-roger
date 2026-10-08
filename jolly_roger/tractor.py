from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations
from pathlib import Path
from time import time
from typing import Any, cast

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
    beam_fraction_to_radius,
    get_baseline_data,
    get_open_ms_tables,
)
from jolly_roger.delays import DelayTime, data_to_delay_time, delay_time_to_data
from jolly_roger.logging import logger
from jolly_roger.plots import plot_baseline_comparison_data, plot_rate_filter_segment
from jolly_roger.rates import (
    ContaminatedSegment,
    RateFilterDiagnostics,
    RateFilterResult,
    RateFilterSettings,
    RateFilterSummary,
    SegmentAccumulator,
    flush_segment_accumulator,
    log_rate_filter_summary,
    rate_filter_segment,
    record_rate_filter_result,
    update_segment_accumulator,
)
from jolly_roger.response import (
    calculate_expected_sinc_width,
    get_delay_of_nth_sidelobe,
)
from jolly_roger.tapering.tukey import get_2d_taper
from jolly_roger.utils import log_dataclass_attributes, log_jolly_roger_version
from jolly_roger.uvws import WDelays, get_object_delay_from_tables, get_w_rates
from jolly_roger.weights import scale_multiple_weights, select_weight_columns
from jolly_roger.wrap import calculate_nyquist_zone, symmetric_domain_wrap


def tukey_taper(
    x: np.typing.NDArray[np.floating[Any]],
    outer_width: float,
    tukey_width: float,
    tukey_x_offset: NDArray[np.floating[Any]] | None = None,
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
        x (np.typing.NDArray[np.floating[Any]]): The intervals to evaluate over. Internally these are concerted to the +/- pi domain
        outer_width (float, optional): The +/- boundary beyond which is 0.0.
        tukey_width (float, optional): Describes the width that the transition from 1.0 to 0.0 occurs.
        tukey_x_offset (NDArray[np.floating[Any]] | None, optional): Sets a new zero point (center of window). Defaults to None.
        notch (bool, optional): Will the taper be used for a notch filter? Defaults to True.

    Returns:
        np.ndarray: The tukey window function
    """
    # Copy to avoid side-effects
    x_local = x.copy()

    if np.any((outer_width - tukey_width) < 0.0):
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
    uvws: NDArray[np.floating[Any]]
    """The uvw coordinates for each loaded data record"""
    time_centroid: NDArray[np.floating[Any]]
    """The time of each data record"""
    ant_1: NDArray[np.int64]
    """Antenna 1 that formed the baseline"""
    ant_2: NDArray[np.int64]
    """Antenna 2 that formed the baseline"""
    row_start: int
    """The starting row of the portion of data loaded"""
    chunk_size: int
    """The size of the data chunk loaded (may be larger if this is the last record)"""
    weights: dict[str, NDArray[np.floating[Any]]] | None = None
    """The weights associated with the data. Key is the column name and the mapped value are the corresponding weights. Only used if reweighting is activated. Defaults to None."""


@dataclass
class DataChunk:
    """Container for a collection of data and associated metadata.
    Here data are drawn from a series of rows.

    Only ``masked_data``, ``freq_chan``, ``time_mjds``, ``ant_1``, ``ant_2`` and
    ``chunk_size`` are consumed by the compute path, so the remaining fields
    carry measurement-set-only bookkeeping and default to sentinels. This lets a
    caller drive the taper directly from arrays without a measurement set.
    """

    masked_data: np.ma.MaskedArray
    """The baseline data, masked where flags are set. shape=(time, chan, pol)"""
    freq_chan: u.Quantity
    """The frequency channels corresponding to the data."""
    time_mjds: NDArray[np.floating[Any]]
    """The raw time extracted from the measurement set in MJDs"""
    ant_1: NDArray[np.int64]
    """The first antenna in the baseline."""
    ant_2: NDArray[np.int64]
    """The second antenna in the baseline."""
    chunk_size: int
    """Size of the chunked portion of the data"""
    phase_center: SkyCoord | None = None
    """The target sky coordinate for the baseline."""
    uvws_phase_center: u.Quantity | None = None
    """The UVW coordinates of the phase center of the baseline."""
    row_start: int = 0
    """Starting row index of the data"""
    weights: dict[str, NDArray[np.floating[Any]]] | None = None
    """The weights associated with the data. Key is the column name and the mapped value are the corresponding weights. Only used if reweighting is activated. Defaults to None."""

    @property
    def time(self) -> Time:
        """The observation times, derived from the raw MJD seconds."""
        return Time(self.time_mjds * u.s, format="mjd", scale="utc")


def _get_data_chunk_from_main_table(
    ms_table: table,
    chunk_size: int,
    data_column: str,
    weight_columns: Sequence[str] | None = None,
) -> Generator[DataChunkArray, None, None]:
    """Return an appropriately size data chunk from the main
    table of a measurement set. These data are ase they are
    in the measurement set without any additional scaling
    or unit adjustments.

    Weights are only returned if the ``weight_column`` is
    set. No distinction is made between a WEIGHT float or
    a WEIGHT spectrum type data. No attempt to validate
    existence of ``weight_column`` is made.

    Args:
        ms_table (table): The opened main table of a measurement set
        chunk_size (int): The size of the data to chunk and return
        data_column (str): The data column to be returned
        weight_columns (Sequence[str] | None, optional): The weight columns to be returned. Defaults to None.

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

        weights: None | dict[str, NDArray[np.floating[Any]]] = None
        if weight_columns:
            weights = {
                weight_column: ms_table.getcol(
                    weight_column, startrow=lower_row, nrow=chunk_size
                )
                for weight_column in weight_columns
            }
            logger.debug(
                f"Getting weights for {weight_columns=} {lower_row} {chunk_size}"
            )

        yield DataChunkArray(
            data=data,
            flags=flags,
            uvws=uvws,
            time_centroid=time_centroid,
            ant_1=ant_1,
            ant_2=ant_2,
            row_start=lower_row,
            chunk_size=chunk_size,
            weights=weights,
        )

        lower_row += chunk_size


def build_data_chunk(
    chunk_array: DataChunkArray,
    freq_chan: u.Quantity,
    phase_dir: SkyCoord,
) -> DataChunk:
    """Attach astropy units/quantities to a raw ``DataChunkArray``.

    Args:
        chunk_array (DataChunkArray): The raw chunk of rows as read from the MS
        freq_chan (u.Quantity): The per-channel frequencies of the MS
        phase_dir (SkyCoord): The phase direction of the MS

    Returns:
        DataChunk: The chunk with units attached
    """
    uvws_phase_center = chunk_array.uvws * u.m
    masked_data = np.ma.masked_array(chunk_array.data, mask=chunk_array.flags)

    return DataChunk(
        masked_data=masked_data,
        freq_chan=freq_chan,
        phase_center=phase_dir,
        uvws_phase_center=uvws_phase_center,
        time_mjds=chunk_array.time_centroid,
        ant_1=chunk_array.ant_1,
        ant_2=chunk_array.ant_2,
        row_start=chunk_array.row_start,
        chunk_size=chunk_array.chunk_size,
        weights=chunk_array.weights,
    )


def get_data_chunks(
    open_ms_tables: OpenMSTables,
    chunk_size: int,
    data_column: str,
    weight_columns: Sequence[str] | None = None,
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
        weight_columns (Sequence[str] | None, optional): The weight columns that would be modified if specified. Defaults to None.

    Yields:
        Generator[DataChunk, None, None]: Representation of the current chunk of rows
    """
    freq_chan = open_ms_tables.spw_table.getcol("CHAN_FREQ").squeeze() * u.Hz
    phase_dir = open_ms_tables.phase_dir

    for data_chunk_array in _get_data_chunk_from_main_table(
        ms_table=open_ms_tables.main_table,
        chunk_size=chunk_size,
        data_column=data_column,
        weight_columns=weight_columns,
    ):
        yield build_data_chunk(
            chunk_array=data_chunk_array, freq_chan=freq_chan, phase_dir=phase_dir
        )


def get_multiple_data_chunks(
    open_ms_tables: OpenMSTables,
    chunk_size: int,
    data_column: str,
    number_of_chunks: int = 1,
    weight_columns: Sequence[str] | None = None,
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
        weight_columns (Sequence[str] | None, optional): The weight columns that would be modified if specified. Defaults to None.

    Yields:
        Generator[tuple[DataChunk, ...], None, None]: Representation of the current chunk of rows
    """
    # To hold the set of data chunks
    base_chunks: list[DataChunk] = []

    # We are using the existing generator. Termination should be
    # straightforward.
    for data_chunk in get_data_chunks(
        open_ms_tables=open_ms_tables,
        chunk_size=chunk_size,
        data_column=data_column,
        weight_columns=weight_columns,
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
) -> bool:
    """Add in the output data column where the modified data
    will be recorded

    Args:
        tab (table): Open reference to the table to modify
        data_column (str, optional): The base data column the new will be based from. Defaults to "DATA".
        output_column (str, optional): The new data column to be created. Defaults to "CORRECTED_DATA".
        overwrite (bool, optional): Whether to overwrite the new output column. Defaults to False.
        copy_column_data (bool, optional): Copy the original data over to the output column. Defaults to False.

    Retuurns:
        bool: Indicates whether output column is empty (True) or contains data (False). This is useful for determining whether to write back to the MS after applying the taper.

    Raises:
        ValueError: Raised if the output column already exists and overwrite is False

    """
    column_is_empty: bool = True
    colnames = tab.colnames()
    if output_column in colnames:
        if not overwrite:
            msg = f"Output column {output_column} already exists in the measurement set. Not overwriting."
            raise ValueError(msg)

        logger.warning(
            f"Output column {output_column} already exists in the measurement set. Will be overwritten!"
        )
        column_is_empty = False
    else:
        logger.info(f"Adding {output_column=}")
        desc = makecoldesc(data_column, tab.getcoldesc(data_column))
        desc["name"] = output_column
        tab.addcols(desc)
        tab.flush()
        column_is_empty = True

    if copy_column_data:
        logger.info(f"Copying {data_column=} to {output_column=}")
        taql(f"UPDATE $tab SET {output_column}={data_column}")
        column_is_empty = False

    return column_is_empty


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
    with table(str(ms_path), readonly=False, ack=False) as tab:
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
    assert max_baselines > 0, f"{max_baselines=}, but should be at least 1"

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

    # Again, note the auto-correlations are mapped to baseline 0!!! Here be pirates mate
    return w_delays.get_indices(
        ant_1=data_chunk.ant_1,
        ant_2=data_chunk.ant_2,
        time_mjds=data_chunk.time_mjds,
    )


@dataclass
class TaperResult:
    """Simple container to help ensure consistent and trackable behaviour
    across levels
    """

    attached_payload: bool = False
    """Indicates whether anything to do."""
    taper: NDArray[np.floating[Any]] | None = None
    """The taper to apply. If None nothing to do."""
    update_flags: bool = False
    """Indicates whether flags need to be updated"""
    flags: NDArray[np.bool_] | None = None
    """The fupdated flags"""
    delay_time: DelayTime | None = None
    """The delay_time taper was constructede against"""
    delay_contaminated: NDArray[np.bool_] | None = None
    """Rows where the object can not be separated from the field in delay"""
    rate_contaminated: NDArray[np.bool_] | None = None
    """Rows where the object can not be separated from the field in delay-rate. Only computed when rate filtering."""


def find_idx_of_closest_delay(x: u.Quantity, object_delays: u.Quantity) -> NDArray[int]:
    """Identify the index in `x` whose element's value is closest
    to the items described in `object_delays`. The `x` remains
    unchanged for all comparison through to `object_delays`.

    Args:
        x (u.Quantity): The array to search for the closest index of (e.g. the delay time spectrum)
        object_delays (u.Quantity): The target values to find matches for

    Returns:
        NDArray[np.int]: The index into `x` where the is closest to values in `object_delays`
    """
    # Attempt to identify the closest idx in time
    diffs = x[:, None] - object_delays[None, :]
    diffs = np.abs(diffs)

    return np.argmin(diffs, axis=0)


def apply_roll_for_taper(
    taper: NDArray[np.floating[Any]], shifts: NDArray[int]
) -> NDArray[np.floating[Any]]:
    """Roll the delay time of each row by a specified number of elements

    The taper should be of shape (no_rows, no_delay_time). The n'th ``shifts``
    will be used to roll the delay time of the n'th row by that value

    Args:
        taper (NDArray[np.floating[Any]]): The taper to modified
        shifts (NDArray[np.int]): The shifts to apply to the delay time for each row

    Returns:
        NDArray[np.floating[Any]]: The shifted taper
    """

    no_rows, no_times = taper.shape
    time_indices = np.arange(no_times)
    # Subtracting shifts moves elements to the right (standard roll behavior)
    shifted_indices = (time_indices - shifts[:, np.newaxis]) % no_times

    # 2. Use advanced indexing to construct the result
    return taper[np.arange(no_rows)[:, np.newaxis], shifted_indices]


def make_search_window(x: u.Quantity, width_ns: float) -> NDArray[np.bool_]:
    """Little helper to test to make sure that the guard window is
    formed with Trues at center"""

    delay_time_ns = x.to("ns").value
    return np.abs(delay_time_ns) < width_ns


def compute_rate_contamination(
    w_delays: WDelays,
    baseline_idx: NDArray[np.int_],
    time_idx: NDArray[np.int_],
    freq_chan: u.Quantity,
    tukey_tractor_options: TukeyTractorOptions,
) -> NDArray[np.bool_]:
    """Identify rows where the object can not be separated from the field in
    delay-rate. The field occupies fringe-rates up to ``nu * rate_guard``, and the
    object sits at ``nu * w_rate``. As both scale with frequency the bands intersect
    when ``|w_rate| <= rate_guard + (guard_hz + width_hz) / nu_min``.

    Args:
        w_delays (WDelays): The object delays and rates
        baseline_idx (NDArray[np.int_]): The baseline index of each row
        time_idx (NDArray[np.int_]): The time index of each row
        freq_chan (u.Quantity): The frequency of each channel
        tukey_tractor_options (TukeyTractorOptions): Options describing the rate filter

    Returns:
        NDArray[np.bool_]: Rows contaminated in delay-rate
    """
    w_rates = np.abs(get_w_rates(w_delays)[baseline_idx, time_idx].value)

    rate_guard = (
        w_delays.rate_guard_region[baseline_idx, time_idx].value
        if w_delays.rate_guard_region is not None
        else np.zeros_like(w_rates)
    )
    # Absolute fringe-rates are converted to a delay-rate at the lowest
    # frequency, where they are the largest
    nu_min_hz = np.min(freq_chan).to(u.Hz).value
    floor_hz = (tukey_tractor_options.rate_filter_guard_hz or 0.0) + (
        tukey_tractor_options.rate_filter_width_hz or 0.0
    )

    return np.asarray(w_rates <= rate_guard + floor_hz / nu_min_hz)


def compute_tukey_taper(
    data_chunk: DataChunk,
    tukey_tractor_options: TukeyTractorOptions,
    w_delays: WDelays,
    delay_time: DelayTime | None = None,
    sidelobe_offset: u.Quantity | None = None,
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
        sidelobe_offset (u.Quantity | None, optional): If provided, the tukey taper will be offset by this amount in delay space in order to target specific sidelobes of the response. Defaults to None.

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
    if isinstance(sidelobe_offset, u.Quantity):
        tukey_x_offset += sidelobe_offset.to(u.s)

    # need to scale the x offset to the -pi to pi (radians) wrap
    # keeping units in seconds though
    # The delay should be symmetric
    tukey_x_offset_sec = symmetric_domain_wrap(
        values=tukey_x_offset.to("s").value,
        upper_limit=np.max(delay_time.delay).to("s").value,
    )

    # Make taper with all units in seconds
    taper = get_2d_taper(
        x=delay_time.delay.to("s").value,
        outer_width=tukey_tractor_options.outer_width_ns * 1e-9,
        tukey_width=tukey_tractor_options.tukey_width_ns * 1e-9,
        tukey_offset=tukey_x_offset_sec,
    )

    # TODO: This pirate reckons that merging the masks together
    # into a single mask throughout may make things easier to
    # manage and visualise.

    # The use of the `tukey_x_offset` changes the
    # shape of the output array. The internals of that
    # function returns a different shape via the broadcasting

    taper = np.swapaxes(taper[:, :, None], 0, 1)
    # taper shape is [chunk_size, no_channels, no_pols]

    stokes_i_delay: NDArray[np.complexfloating[Any]] | None = None
    if tukey_tractor_options.peak_shift_search:
        # The formed stokes I spectrum code be reused later should
        # comparisons to field or minimum flux cuts are applied
        stokes_i_delay = np.sum(np.abs(delay_time.delay_time[..., [0, 3]]), axis=-1)
        # We have to get the closest element from the time and not the minimum of
        # the taper as the taper can have a 'top hat' zero region
        object_idx = find_idx_of_closest_delay(
            x=delay_time.delay, object_delays=tukey_x_offset_sec * u.s
        )

        if tukey_tractor_options.peak_shift_search_width_ns is None:
            # This isolates the spectrum of the source in delay space
            inverted_taper = (1.0 - taper)[..., 0]

            # Here the taper is used to isolate the objects spectrum, and
            # then we look for the peak. The taper should be reasonably well
            # constructed to be in approximately the right location
            object_response = inverted_taper * stokes_i_delay

        else:
            # a strict window has been provided that will be used to search
            # around the object. We will construct the window, roll it to
            # expected object position, then roll t he taper
            search_mask = make_search_window(
                x=delay_time.delay,
                width_ns=tukey_tractor_options.peak_shift_search_width_ns,
            )
            # Make the mask match t he shape of the data chunk. This is fine as the search width is
            # constant across all baselines/dimensions (unlike the guard zone)
            search_mask = np.broadcast_to(
                search_mask, (object_idx.size, search_mask.size)
            )
            # Rhw object idx is relative to the 0th element, but
            # the search mask is orientated around 0-seconds, which could be anywhere
            zero_idx = find_idx_of_closest_delay(
                x=delay_time.delay,
                object_delays=np.zeros_like(tukey_x_offset_sec) * u.s,
            )
            # Now roll the search window
            search_mask = apply_roll_for_taper(
                taper=search_mask, shifts=object_idx - zero_idx
            )
            # The mask should not be at the the object predicted position
            object_response = search_mask * stokes_i_delay

        # Now find the peak response and determine the shift
        peak_idx = np.argmax(object_response, axis=1)
        shifts = peak_idx - object_idx
        logger.info(f"{shifts=}")

        taper = apply_roll_for_taper(taper=taper[..., 0], shifts=shifts)[..., None]

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
    field_outer_width = tukey_tractor_options.outer_width_ns * 1e-9
    if w_delays.guard_region is not None:
        field_outer_width += w_delays.guard_region[baseline_idx, time_idx].to("s").value

    field_taper = get_2d_taper(
        x=delay_time.delay.to("s").value,
        outer_width=field_outer_width,
        tukey_width=tukey_tractor_options.tukey_width_ns * 1e-9,
        tukey_offset=None,
    )
    # field_taper.shape is [no_channels, ]
    # We need to account for no broadcasting when offset is None
    # as the returned shape is different
    field_taper = np.swapaxes(field_taper[:, :, None], 0, 1)
    intersecting_taper = np.any(
        np.reshape((taper != 1) & (field_taper != 1), (taper.shape[0], -1)), axis=1
    )

    # Here we consider what to do if want to compare brightness of the object in delay
    # space is less than that of the field. If the object is not detected we ought to
    # set the tape to 1 so the data are not modified
    if (
        tukey_tractor_options.compare_to_field is not None
        or tukey_tractor_options.object_minimum_flux is not None
    ):
        # The delay spectrum are complex quantities, and we need to compare
        # the flux
        # Make a stokes I type spectrum

        if stokes_i_delay is None:
            stokes_i_delay = np.abs(
                np.sum(delay_time.delay_time[..., [0, -1]], axis=-1)
            )

        _field_taper = np.squeeze(field_taper)
        field_stats = np.max(stokes_i_delay * (1.0 - _field_taper), axis=1)
        object_stats = np.max(stokes_i_delay * (1.0 - taper[..., 0]), axis=1)

        if tukey_tractor_options.compare_to_field is not None:
            flux_mask = (
                object_stats < tukey_tractor_options.compare_to_field * field_stats
            )

            # For any element where there is not enough flux set the taper so
            # it does not modify the data
            taper[flux_mask, :] = 1.0
        if tukey_tractor_options.object_minimum_flux is not None:
            min_flux_mask = object_stats < tukey_tractor_options.object_minimum_flux
            taper[min_flux_mask, :] = 1.0

    # # Should the data need to be modified in conjunction with the flags
    # taper[
    #     intersecting_taper &
    #     ~elevation_mask[time_idx] &
    #     ~ignore_wrapping_for
    # ] = 0.0
    rate_contaminated: NDArray[np.bool_] | None = None
    if tukey_tractor_options.rate_filter:
        rate_contaminated = compute_rate_contamination(
            w_delays=w_delays,
            baseline_idx=baseline_idx,
            time_idx=time_idx,
            freq_chan=data_chunk.freq_chan,
            tukey_tractor_options=tukey_tractor_options,
        )

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
        delay_contaminated=intersecting_taper,
        rate_contaminated=rate_contaminated,
    )


def apply_taper(
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
class RateFilterPayload:
    """The per-row information needed to delay-rate filter rows of a chunk"""

    original_masked_data: np.ma.MaskedArray
    """The visibilities before any delay tapering"""
    baseline_idx: NDArray[np.int_]
    """The index of each row's baseline into the ``WDelays``"""
    time_idx: NDArray[np.int_]
    """The index of each row's time into the ``WDelays``"""
    delay_contaminated: NDArray[np.bool_]
    """Rows contaminated in delay by any object"""
    recoverable: NDArray[np.bool_]
    """Rows contaminated in delay where every contaminating object is separable in delay-rate"""
    original_weights: dict[str, NDArray[np.floating[Any]]] | None = None
    """The weights before any scaling"""


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
    flags: NDArray[np.bool_] | None = None
    """The flags to write back"""
    update_weights: bool = False
    """Indicates whether data should be written back to the MS"""
    weights: dict[str, NDArray[np.floating[Any]]] | None = None
    """The scaled weights that should be written back to the MS. The key is the column name and the mapped values are the corresponding scaled weights. If None nothing to write back."""
    rate_payload: RateFilterPayload | None = None
    """Information to delay-rate filter contaminated rows. Only set when rate filtering."""


def make_rate_filter_payload(
    data_chunk: DataChunk,
    w_delays: WDelays,
    taper_results: list[TaperResult],
) -> RateFilterPayload:
    """Combine the per-object contamination of rows. A row is recoverable when it
    is contaminated in delay by some object, and no object contaminating it in delay
    is also contaminated in delay-rate.

    Args:
        data_chunk (DataChunk): The chunk before any tapering
        w_delays (WDelays): Any of the objects, used for their baseline and time mappings
        taper_results (list[TaperResult]): The results of each object (and sidelobe)

    Returns:
        RateFilterPayload: The information needed to delay-rate filter the chunk
    """
    baseline_idx, time_idx = _get_baseline_time_indicies(
        w_delays=w_delays, data_chunk=data_chunk
    )
    n_rows = len(data_chunk.ant_1)

    delay_contaminated = np.zeros(n_rows, dtype=bool)
    doubly_contaminated = np.zeros(n_rows, dtype=bool)
    for taper_result in taper_results:
        if taper_result.delay_contaminated is None:
            continue
        delay_contaminated |= taper_result.delay_contaminated
        if taper_result.rate_contaminated is not None:
            doubly_contaminated |= (
                taper_result.delay_contaminated & taper_result.rate_contaminated
            )

    # Auto-correlations are mapped onto another baseline, so are not filtered
    cross_correlation = data_chunk.ant_1 != data_chunk.ant_2
    recoverable = delay_contaminated & ~doubly_contaminated & cross_correlation

    logger.debug(
        f"Rows contaminated in delay: {np.sum(delay_contaminated)}, "
        f"in delay and delay-rate: {np.sum(doubly_contaminated)}, "
        f"recoverable: {np.sum(recoverable)}"
    )

    return RateFilterPayload(
        original_masked_data=data_chunk.masked_data,
        baseline_idx=baseline_idx,
        time_idx=time_idx,
        delay_contaminated=delay_contaminated,
        recoverable=recoverable,
        original_weights=data_chunk.weights,
    )


def compute_tukey_multi_taper(
    data_chunk: DataChunk,
    tukey_tractor_options: TukeyTractorOptions,
    w_delays_list: list[WDelays],
) -> TaperedChunkResult:
    chunk_size = data_chunk.chunk_size

    outer_width = tukey_tractor_options.outer_width_ns * 1e-9 * u.s

    # Get the results for each object. Note reusing the delay_time object
    # to avoid unnecessary recomputes. Also why the for loop and not list comphrension
    delay_time: DelayTime | None = None
    taper_results = []
    for w_delays in w_delays_list:
        taper_result = compute_tukey_taper(
            data_chunk=data_chunk,
            tukey_tractor_options=tukey_tractor_options,
            w_delays=w_delays,
            delay_time=delay_time,
        )
        # An object that is skipped (e.g. below the elevation cut) has no
        # delay_time, and should not discard the one formed for an earlier object
        if taper_result.delay_time is not None:
            delay_time = taper_result.delay_time
        taper_results.append(taper_result)

        if (
            not taper_result.attached_payload
            or tukey_tractor_options.nth_sidelobe_null is None
        ):
            continue

        # If sidelobe nulling is used than the outer_width has been configured
        # by the auto-size option
        for sign in (1, -1):
            for sidelobe in range(1, tukey_tractor_options.nth_sidelobe_null + 1):
                sidelobe_offset = get_delay_of_nth_sidelobe(
                    n=sidelobe, sinc_width=outer_width
                )
                taper_result = compute_tukey_taper(
                    data_chunk=data_chunk,
                    tukey_tractor_options=tukey_tractor_options,
                    w_delays=w_delays,
                    delay_time=delay_time,
                    sidelobe_offset=sidelobe_offset * sign,
                )
                if taper_result.delay_time is not None:
                    delay_time = taper_result.delay_time
                taper_results.append(taper_result)

    # Note that applying the taper replaces (rather than modifies) the masked
    # data of the data chunk, so a reference keeps the original visibilities
    rate_payload: RateFilterPayload | None = None
    if tukey_tractor_options.rate_filter:
        rate_payload = make_rate_filter_payload(
            data_chunk=data_chunk,
            w_delays=w_delays_list[0],
            taper_results=taper_results,
        )

    # Handle all the cases. If all data chunks showed nothing to do we can
    # return early and provide the original data back to the caller.
    if all(not taper_result.attached_payload for taper_result in taper_results):
        return TaperedChunkResult(
            chunk_size=chunk_size,
            nothing_to_do=True,
            data_chunk=data_chunk,
            rate_payload=rate_payload,
        )

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

    tapered_data = apply_taper(
        data_chunk=data_chunk,
        delay_time=cast(DelayTime, delay_time),
        taper=combined_taper,
    )

    # Should weights be provided to the DataChunk than it is assumed that they
    # need to be scaleded based on the final taper. The driving functions that
    # call into this multi tractor are responsible to writing this back out to
    # the appropriate coluumn
    scaled_weights: None | dict[str, NDArray[np.floating[Any]]] = None
    update_weights = False
    if data_chunk.weights is not None:
        scaled_weights = scale_multiple_weights(
            taper=combined_taper, weights=data_chunk.weights
        )

        # Should there actually be weights attached than at this point
        # in the code path there is a taper that has been applied.
        update_weights = True

    return TaperedChunkResult(
        chunk_size=chunk_size,
        nothing_to_do=False,
        update_data=True,
        data_chunk=tapered_data,
        update_flags=update_flags,
        flags=combined_flags,
        update_weights=update_weights,
        weights=scaled_weights,
        rate_payload=rate_payload,
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
    auto_size: bool = False
    """Automatically size the outer width of the tukey taper based on the data. With rate_filter, the delay-rate width of each segment is also sized from its duration T as (N+1)/T, reduced towards 1/T as needed to keep the object separable from the field. Overrides outer_width_ns, tukey_width_ns and rate_filter_width_hz."""
    nth_sidelobe_null: int | None = None
    """Null up to the N'th sidelobe. Only used in auto_size mode. With rate_filter this is also the number of delay-rate sidelobes (N) included, where None includes 1. Defaults to None."""
    reweight: bool = False
    """Attempt to identify a WEIGHT-like column and rescale to indicate modified data. Defaults to False."""
    weight_column: str | None = None
    """The name of the WEIGHT-like column. If None when rewrite is True the WEIGHT-like column will be searched for. Defaults to None."""
    guard_field: bool = False
    """If True derive a region around the delay=0 spectrum to protect the field-of-view/"""
    guard_field_fraction: float = 0.1
    """The attenuation level of the main lobe to guard to, and should be in the range (0, 1). Values closer to zero correspond to a larger field-of-view, and hence a larger guard band in delay space. """
    object_minimum_flux: float | None = None
    """The minimum absolute flux an object should have (as measured in delay space in Jy) for it to be nulled"""
    peak_shift_search: bool = False
    """Search around the predicted delay for the peak in the delay spectrum to account for shifts (e.g. ionspheric shift, inaccuracies in prediction). """
    peak_shift_search_width_ns: float | None = None
    """If provided this will be used to set strong limits to search for a peak around a predicted objects position. If None when peak search is activated, the taper is used in stead."""
    rate_filter: bool = False
    """Filter in delay and delay-rate the timesteps where an object is contaminated in delay but separable in delay-rate. Segments are filtered once the object leaves the contaminated zone."""
    rate_filter_width_hz: float | None = None
    """The width beyond the object's predicted fringe-rate band over which the delay-rate taper rolls off (1 - cos) from zero to one, in Hz. If None two rate bins are used. Overridden by auto_size."""
    rate_filter_guard_hz: float | None = None
    """A fringe-rate around zero to protect, added to the guard derived from the field-of-view (see ``guard_field``). If None one rate bin is used."""
    rate_filter_min_timesteps: int = 8
    """The minimum number of contaminated timesteps of a segment for it to be delay-rate filtered"""
    rate_filter_max_timesteps: int | None = None
    """The maximum number of contaminated timesteps collected for a baseline before it is delay-rate filtered. Limits memory usage. If None there is no limit."""
    rate_filter_pad_timesteps: int = 0
    """The number of clean timesteps either side of a contaminated segment to include when delay-rate filtering. These are not modified. A value of 0 disables padding."""
    unflag_rate_filtered: bool = False
    """Remove the contamination flags of rows that were delay-rate filtered"""
    rate_filter_plots: bool = False
    """Plot the delay vs delay-rate of each segment that is delay-rate filtered"""
    rate_filter_max_plots: int = 20
    """The maximum number of delay-rate filtered segments to plot"""


@dataclass(frozen=True)
class TukeyTractorResults:
    """Simple return set of results from the tractoring process"""

    ms_path: Path
    """Path to the measurement set that was modified"""
    output_column: str
    """The name of the column that has the modified/tapered visibilities"""
    output_plots: list[Path] | None = None
    """The output plots that were created, if any"""


def compute_auto_taper_widths(
    freq_chan: u.Quantity, tukey_tractor_options: TukeyTractorOptions
) -> TukeyTractorOptions:
    """Derive the tukey taper widths from the expected size of the sinc
    function for the given channel frequencies.

    The delay-rate width depends on the duration of each segment, so it is not
    set here. Instead ``rate_filter_width_hz`` is cleared, and each segment is
    sized from its own expected sinc response in delay-rate.

    Args:
        freq_chan (u.Quantity): The per-channel frequencies of the MS
        tukey_tractor_options (TukeyTractorOptions): Tukey tractor options to inspect

    Returns:
        TukeyTractorOptions: Duplicate options as input, expect the specification of the taper is overwritten
    """
    sinc_width = calculate_expected_sinc_width(freqs=freq_chan)
    outer_width_ns = sinc_width.to("ns").value
    logger.info(f"Setting automatic {outer_width_ns=}")
    if tukey_tractor_options.rate_filter:
        n_sidelobes = tukey_tractor_options.nth_sidelobe_null or 1
        logger.info(
            f"Setting automatic delay-rate width per segment: ({n_sidelobes}+1)/T, fitted down to 1/T"
        )

    return tukey_tractor_options.with_options(
        outer_width_ns=outer_width_ns,
        tukey_width_ns=0.0,  # type: ignore[arg-type]
        rate_filter_width_hz=None,  # type: ignore[arg-type]
    )
    # TODO: Removing the type ignore above results in a mypy error in capn_crunch


def _set_auto_taper_widths(
    open_ms_tables: OpenMSTables, tukey_tractor_options: TukeyTractorOptions
) -> TukeyTractorOptions:
    """Obtain the expected size of the sinc function for the spectral information
    contained in the MS

    Args:
        open_ms_tables (OpenMSTables): Collection of MS tables to inspected
        tukey_tractor_options (TukeyTractorOptions): Tukey tractor options to inspect

    Returns:
        TukeyTractorOptions: Duplicate options as input, expect the specification of the taper is overwritten
    """
    freq_chan = open_ms_tables.spw_table.getcol("CHAN_FREQ").squeeze() * u.Hz
    return compute_auto_taper_widths(
        freq_chan=freq_chan, tukey_tractor_options=tukey_tractor_options
    )


def write_back_results(
    open_ms_tables: OpenMSTables,
    taper_chunk_result: TaperedChunkResult,
    tukey_tractor_options: TukeyTractorOptions,
    write_back_required: bool,
    pbar: tqdm | None = None,
) -> None:
    """Write back data chunk results to the measurement set

    Args:
        open_ms_tables (OpenMSTables): The set of open handlers to the relevant measurement sets
        taper_chunk_result (TaperedChunkResult): The collect of results to consider around writing back
        tukey_tractor_options (TukeyTractorOptions): Options relevant to the data selection
        pbar (tqdm): A handler to the progress bar
        write_back_required (bool): Whether a write back is required in all cases
    """
    if pbar is not None:
        pbar.update(taper_chunk_result.chunk_size)

    if taper_chunk_result.nothing_to_do and not write_back_required:
        logger.debug("No attached data payload, skipping")
        return

    data_chunk = taper_chunk_result.data_chunk
    assert data_chunk is not None, f"{data_chunk=}, which should not happen"

    # Only update here is we pass the dry run check above. If data are not copied
    # upfront for new column than this is necessary
    if taper_chunk_result.update_data or write_back_required:
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
        logger.debug("No updating of flags required, skipping for chunk")
    if taper_chunk_result.update_weights:
        logger.debug("Updating weights")
        assert taper_chunk_result.weights is not None, (
            "Expected weights to be attached, found None"
        )
        for (
            weight_column_name,
            scaled_weights,
        ) in taper_chunk_result.weights.items():
            open_ms_tables.main_table.putcol(
                columnname=weight_column_name,
                value=scaled_weights,
                startrow=data_chunk.row_start,
                nrow=data_chunk.chunk_size,
            )
    else:
        logger.debug("No updating of weights")


def accumulate_rate_filter_rows(
    accumulator: SegmentAccumulator, taper_chunk_result: TaperedChunkResult
) -> list[ContaminatedSegment]:
    """Feed the rows of a processed chunk to the accumulator

    Args:
        accumulator (SegmentAccumulator): Per-baseline collection of contaminated rows
        taper_chunk_result (TaperedChunkResult): The processed chunk, with its rate filter payload

    Returns:
        list[ContaminatedSegment]: Segments that are ready to be filtered
    """
    payload = taper_chunk_result.rate_payload
    data_chunk = taper_chunk_result.data_chunk
    assert payload is not None, "Rate filter payload expected"
    assert data_chunk is not None, "Data chunk expected"

    original = payload.original_masked_data
    return update_segment_accumulator(
        accumulator=accumulator,
        row_numbers=data_chunk.row_start + np.arange(len(data_chunk.ant_1)),
        ant_1=data_chunk.ant_1,
        ant_2=data_chunk.ant_2,
        baseline_idx=payload.baseline_idx,
        time_mjds=data_chunk.time_mjds,
        time_idx=payload.time_idx,
        data=np.ma.getdata(original),
        mask=np.ma.getmaskarray(original),
        recoverable=payload.recoverable,
        delay_contaminated=payload.delay_contaminated,
        weights=payload.original_weights,
    )


def write_rate_filtered_segment(
    open_ms_tables: OpenMSTables,
    rate_filter_result: RateFilterResult,
    tukey_tractor_options: TukeyTractorOptions,
) -> None:
    """Write the delay-rate filtered rows of a segment back to the measurement set

    Args:
        open_ms_tables (OpenMSTables): The set of open handlers to the relevant measurement sets
        rate_filter_result (RateFilterResult): The filtered segment
        tukey_tractor_options (TukeyTractorOptions): Options relevant to the data selection
    """
    if not rate_filter_result.success:
        logger.debug(
            f"Segment of {len(rate_filter_result.rows)} rows not filtered: {rate_filter_result.reason}"
        )
        return

    assert rate_filter_result.data is not None, "Filtered data expected"
    with open_ms_tables.main_table.selectrows(rate_filter_result.rows) as subtab:
        subtab.putcol(tukey_tractor_options.output_column, rate_filter_result.data)
        if (
            tukey_tractor_options.unflag_rate_filtered
            and rate_filter_result.flags is not None
        ):
            subtab.putcol("FLAG", rate_filter_result.flags)
        if rate_filter_result.weights is not None:
            for (
                weight_column_name,
                scaled_weights,
            ) in rate_filter_result.weights.items():
                subtab.putcol(weight_column_name, scaled_weights)


def merge_rate_filter_results(
    rate_filter_results: Sequence[RateFilterResult],
) -> RateFilterResult:
    """Combine filtered segments into a single result whose rows are sorted.
    Writing few, large, ordered selections is far faster than many small
    selections, particularly for tiled columns.

    Args:
        rate_filter_results (Sequence[RateFilterResult]): Successfully filtered segments

    Returns:
        RateFilterResult: The combined segments, sorted by row
    """
    assert len(rate_filter_results) > 0, "Expected at least one result to merge"
    assert all(result.success for result in rate_filter_results), (
        "Only filtered segments may be merged"
    )

    rows = np.concatenate([result.rows for result in rate_filter_results])
    order = np.argsort(rows, kind="stable")

    def _merge(arrays: Sequence[NDArray[Any]]) -> NDArray[Any]:
        return np.concatenate(arrays)[order]

    data = [result.data for result in rate_filter_results if result.data is not None]
    assert len(data) == len(rate_filter_results), "Filtered data expected"
    flags = [result.flags for result in rate_filter_results if result.flags is not None]
    weights = [
        result.weights for result in rate_filter_results if result.weights is not None
    ]

    return RateFilterResult(
        rows=rows[order],
        success=True,
        data=_merge(data),
        flags=_merge(flags) if len(flags) == len(rate_filter_results) else None,
        weights={k: _merge([w[k] for w in weights]) for k in weights[0]}
        if len(weights) == len(rate_filter_results)
        else None,
    )


@dataclass
class RateFilterWriteBuffer:
    """Delay-rate filtered segments waiting to be written to the measurement set in
    large, row-ordered batches. Segments are added with ``add_to_rate_filter_write_buffer``."""

    open_ms_tables: OpenMSTables
    """The set of open handlers to the relevant measurement sets"""
    tukey_tractor_options: TukeyTractorOptions
    """Options relevant to the data selection"""
    max_rows: int
    """The number of buffered rows that triggers a write"""
    results: list[RateFilterResult] = field(default_factory=list)
    """The buffered, successfully filtered, segments"""
    n_rows: int = 0
    """The number of buffered rows"""


def flush_rate_filter_write_buffer(write_buffer: RateFilterWriteBuffer) -> None:
    """Write all buffered segments to the measurement set

    Args:
        write_buffer (RateFilterWriteBuffer): The buffered segments, emptied in place
    """
    if not write_buffer.results:
        return

    logger.debug(
        f"Writing {len(write_buffer.results)} delay-rate filtered segments ({write_buffer.n_rows} rows)"
    )
    write_rate_filtered_segment(
        open_ms_tables=write_buffer.open_ms_tables,
        rate_filter_result=merge_rate_filter_results(write_buffer.results),
        tukey_tractor_options=write_buffer.tukey_tractor_options,
    )
    write_buffer.results = []
    write_buffer.n_rows = 0


def add_to_rate_filter_write_buffer(
    write_buffer: RateFilterWriteBuffer, rate_filter_result: RateFilterResult
) -> None:
    """Buffer a filtered segment, writing the buffer once it is full. Segments
    that were not filtered are ignored.

    Args:
        write_buffer (RateFilterWriteBuffer): The buffered segments, updated in place
        rate_filter_result (RateFilterResult): The outcome of filtering a segment
    """
    if not rate_filter_result.success:
        logger.debug(
            f"Segment of {len(rate_filter_result.rows)} rows not filtered: {rate_filter_result.reason}"
        )
        return

    write_buffer.results.append(rate_filter_result)
    write_buffer.n_rows += len(rate_filter_result.rows)
    if write_buffer.n_rows >= write_buffer.max_rows:
        flush_rate_filter_write_buffer(write_buffer)


def make_rate_filter_plot_path(
    ms_path: Path, diagnostics: RateFilterDiagnostics
) -> Path:
    """The output path of a delay-rate filtered segment's plot, placed alongside other plots

    Args:
        ms_path (Path): The measurement set being processed
        diagnostics (RateFilterDiagnostics): The filtered segment

    Returns:
        Path: Location to save the plot to
    """
    output_dir = ms_path.parent / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    return (
        output_dir
        / f"{ms_path.name}_rate_filter_{diagnostics.ant_1}_{diagnostics.ant_2}_row{diagnostics.first_row}.png"
    )


def _rate_filter_settings(
    tukey_tractor_options: TukeyTractorOptions,
) -> RateFilterSettings:
    """Extract the options needed to delay-rate filter a segment"""
    return RateFilterSettings(
        outer_width_ns=tukey_tractor_options.outer_width_ns,
        tukey_width_ns=tukey_tractor_options.tukey_width_ns,
        width_hz=tukey_tractor_options.rate_filter_width_hz,
        guard_hz=tukey_tractor_options.rate_filter_guard_hz,
        min_timesteps=tukey_tractor_options.rate_filter_min_timesteps,
        elevation_cut_deg=tukey_tractor_options.elevation_cut_deg,
        ignore_nyquist_zone=tukey_tractor_options.ignore_nyquist_zone,
        auto_width=tukey_tractor_options.auto_size,
        auto_sidelobes=tukey_tractor_options.nth_sidelobe_null or 1,
    )


@dataclass
class RateFilterProcessor:
    """State to delay-rate filter the contaminated rows of a measurement set as it
    is processed chunk by chunk. Rows are collected per-baseline, and each segment
    is filtered, optionally plotted, and written back once it is released."""

    open_ms_tables: OpenMSTables
    """The set of open handlers to the measurement set being processed"""
    tukey_tractor_options: TukeyTractorOptions
    """Options describing the rate filter"""
    w_delays_list: list[WDelays]
    """The objects to null"""
    settings: RateFilterSettings
    """The subset of options needed to filter a segment"""
    freq_chan: u.Quantity
    """The frequency of each channel"""
    accumulator: SegmentAccumulator
    """Per-baseline collection of contaminated rows"""
    write_buffer: RateFilterWriteBuffer
    """Filtered segments waiting to be written back"""
    summary: RateFilterSummary = field(default_factory=RateFilterSummary)
    """Tally of the filtering outcomes"""
    plot_paths: list[Path] = field(default_factory=list)
    """The plots of filtered segments made so far"""


def make_rate_filter_processor(
    open_ms_tables: OpenMSTables,
    tukey_tractor_options: TukeyTractorOptions,
    w_delays_list: list[WDelays],
) -> RateFilterProcessor:
    """Set up the delay-rate filtering of a measurement set

    Args:
        open_ms_tables (OpenMSTables): The set of open handlers to the measurement set being processed
        tukey_tractor_options (TukeyTractorOptions): Options describing the rate filter
        w_delays_list (list[WDelays]): The objects to null

    Returns:
        RateFilterProcessor: State used by ``process_rate_filter_chunk`` and ``finish_rate_filter``
    """
    return RateFilterProcessor(
        open_ms_tables=open_ms_tables,
        tukey_tractor_options=tukey_tractor_options,
        w_delays_list=w_delays_list,
        settings=_rate_filter_settings(tukey_tractor_options),
        freq_chan=open_ms_tables.spw_table.getcol("CHAN_FREQ").squeeze() * u.Hz,
        accumulator=SegmentAccumulator(
            max_timesteps=tukey_tractor_options.rate_filter_max_timesteps,
            pad_timesteps=tukey_tractor_options.rate_filter_pad_timesteps,
        ),
        # Rows of a released segment are all in chunks that have already been
        # written, so deferring the segment writes can not be overwritten later
        write_buffer=RateFilterWriteBuffer(
            open_ms_tables=open_ms_tables,
            tukey_tractor_options=tukey_tractor_options,
            max_rows=tukey_tractor_options.chunk_size
            * tukey_tractor_options.max_workers,
        ),
    )


def _rate_filter_plot_wanted(rate_filter_processor: RateFilterProcessor) -> bool:
    """Whether the next filtered segment should be plotted"""
    options = rate_filter_processor.tukey_tractor_options
    return (
        options.rate_filter_plots
        and len(rate_filter_processor.plot_paths) < options.rate_filter_max_plots
    )


def _filter_rate_segments(
    rate_filter_processor: RateFilterProcessor,
    segments: Sequence[ContaminatedSegment],
) -> None:
    """Filter, optionally plot, and buffer the writing of released segments"""
    for segment in segments:
        rate_filter_result = rate_filter_segment(
            segment=segment,
            freq_chan=rate_filter_processor.freq_chan,
            w_delays_list=rate_filter_processor.w_delays_list,
            settings=rate_filter_processor.settings,
            keep_diagnostics=_rate_filter_plot_wanted(rate_filter_processor),
        )
        record_rate_filter_result(rate_filter_processor.summary, rate_filter_result)
        if rate_filter_result.diagnostics is not None:
            rate_filter_processor.plot_paths.append(
                plot_rate_filter_segment(
                    diagnostics=rate_filter_result.diagnostics,
                    output_path=make_rate_filter_plot_path(
                        ms_path=rate_filter_processor.open_ms_tables.ms_path,
                        diagnostics=rate_filter_result.diagnostics,
                    ),
                )
            )
        add_to_rate_filter_write_buffer(
            rate_filter_processor.write_buffer, rate_filter_result
        )


def process_rate_filter_chunk(
    rate_filter_processor: RateFilterProcessor,
    taper_chunk_result: TaperedChunkResult,
) -> None:
    """Collect the rows of a chunk, and filter any segments that are released.
    Must be called in row order, after the chunk is written back.

    Args:
        rate_filter_processor (RateFilterProcessor): The delay-rate filtering state
        taper_chunk_result (TaperedChunkResult): The processed chunk, with its rate filter payload
    """
    _filter_rate_segments(
        rate_filter_processor=rate_filter_processor,
        segments=accumulate_rate_filter_rows(
            accumulator=rate_filter_processor.accumulator,
            taper_chunk_result=taper_chunk_result,
        ),
    )


def finish_rate_filter(rate_filter_processor: RateFilterProcessor) -> None:
    """Filter the remaining segments, write everything back and log a summary

    Args:
        rate_filter_processor (RateFilterProcessor): The delay-rate filtering state
    """
    _filter_rate_segments(
        rate_filter_processor=rate_filter_processor,
        segments=flush_segment_accumulator(rate_filter_processor.accumulator),
    )
    flush_rate_filter_write_buffer(rate_filter_processor.write_buffer)
    log_rate_filter_summary(
        rate_filter_processor.summary,
        doubly_contaminated_rows=rate_filter_processor.accumulator.doubly_contaminated_rows,
    )
    if rate_filter_processor.plot_paths:
        logger.info(
            f"Made {len(rate_filter_processor.plot_paths)} delay-rate filter plots in {rate_filter_processor.plot_paths[0].parent}"
        )


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

    if (
        not tukey_tractor_options.auto_size
        and tukey_tractor_options.nth_sidelobe_null is not None
    ):
        logger.warning(
            "nth_sidelobe_null is only used in auto_size mode. Since auto_size is False, nth_sidelobe_null will be ignored."
        )
        tukey_tractor_options = tukey_tractor_options.with_options(
            nth_sidelobe_null=None,  # type: ignore[arg-type]
        )

    if tukey_tractor_options.rate_filter and not tukey_tractor_options.guard_field:
        logger.warning(
            "rate_filter is set without guard_field. Only rate_filter_guard_hz protects the field in delay-rate."
        )

    # acquire all the tables necessary to get unit information and data from
    open_ms_tables = get_open_ms_tables(ms_path=ms_path, read_only=False)

    if tukey_tractor_options.auto_size:
        tukey_tractor_options = _set_auto_taper_widths(
            open_ms_tables=open_ms_tables, tukey_tractor_options=tukey_tractor_options
        )

    # Weights are only read, scaled and written back when reweighting is requested
    weight_columns: Sequence[str] | None = None
    if tukey_tractor_options.reweight:
        weight_columns = select_weight_columns(
            ms_path=ms_path, weight_column=tukey_tractor_options.weight_column
        )
    elif tukey_tractor_options.weight_column is not None:
        logger.warning(
            f"{tukey_tractor_options.weight_column=} is set but reweight is False. Weights will not be modified."
        )

    write_back_required: bool = True
    if not tukey_tractor_options.dry_run:
        # Data will need to be written back to the MS after each chunk if the
        # output column has not data
        write_back_required = add_output_column(
            tab=open_ms_tables.main_table,
            output_column=tukey_tractor_options.output_column,
            data_column=tukey_tractor_options.data_column,
            overwrite=tukey_tractor_options.overwrite,
            copy_column_data=tukey_tractor_options.copy_column_data,
        )

    # Calculate the guard field region used to derieve the appropriate delay window
    radial_fov: u.Quantity | None = None
    if tukey_tractor_options.guard_field:
        radial_fov = beam_fraction_to_radius(
            fraction=tukey_tractor_options.guard_field_fraction,
            field_of_view=open_ms_tables.nominal_fov,
        )

    # Generate the delay for all baselines and time steps. Reuse the already-open
    # tables so the MS is not opened a second time.
    w_delays_list = get_object_delay_from_tables(
        open_ms_tables=open_ms_tables,
        phase_dir=open_ms_tables.phase_dir,
        object_name=tukey_tractor_options.target_objects,
        reverse_baselines=tukey_tractor_options.reverse_baselines,
        flip_uvw_sign=tukey_tractor_options.flip_uvw_sign,
        radial_fov=radial_fov,
    )
    assert all(len(w_delays.w_delays.shape) == 2 for w_delays in w_delays_list), (
        "Sanity check failed, incorrect dimensionality returned"
    )

    if not tukey_tractor_options.dry_run:
        pool: None | ThreadPoolExecutor = None
        if tukey_tractor_options.max_workers > 1:
            logger.info(
                f"Starting {tukey_tractor_options.max_workers} compute workers. Be mindful of {tukey_tractor_options.chunk_size=}"
            )
            pool = ThreadPoolExecutor(max_workers=tukey_tractor_options.max_workers)

        # This could be offloaded to some other function if we end up
        # havig multiple modes
        partial_compute_func = partial(
            compute_tukey_multi_taper,
            tukey_tractor_options=tukey_tractor_options,
            w_delays_list=w_delays_list,
        )

        rate_filter_processor: RateFilterProcessor | None = None
        if tukey_tractor_options.rate_filter:
            rate_filter_processor = make_rate_filter_processor(
                open_ms_tables=open_ms_tables,
                tukey_tractor_options=tukey_tractor_options,
                w_delays_list=w_delays_list,
            )

        logger.info(f"Incremental data flushes {write_back_required=}")
        start = time()
        total_tukey_time_s = 0.0
        with tqdm(total=len(open_ms_tables.main_table), desc="Rows") as pbar:
            for data_chunk_tuple in get_multiple_data_chunks(
                open_ms_tables=open_ms_tables,
                chunk_size=tukey_tractor_options.chunk_size,
                data_column=tukey_tractor_options.data_column,
                number_of_chunks=tukey_tractor_options.max_workers,
                weight_columns=weight_columns,
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
                    write_back_results(
                        open_ms_tables=open_ms_tables,
                        taper_chunk_result=taper_chunk_result,
                        tukey_tractor_options=tukey_tractor_options,
                        write_back_required=write_back_required,
                        pbar=pbar,
                    )

                    # Segments overwrite rows already written above
                    if rate_filter_processor is not None:
                        process_rate_filter_chunk(
                            rate_filter_processor=rate_filter_processor,
                            taper_chunk_result=taper_chunk_result,
                        )

            if rate_filter_processor is not None:
                finish_rate_filter(rate_filter_processor=rate_filter_processor)

        stop = time()
        runtime_s = stop - start
        assert data_chunk_tuple[-1] is not None, "data_chunk is not formed correctly"
        logger.info(
            f"Tapered {len(tukey_tractor_options.target_objects)} targets over {len(open_ms_tables.main_table)} rows by {len(data_chunk_tuple[-1].freq_chan)} chans in {runtime_s:0.2f}s"
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
