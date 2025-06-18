from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from astropy.coordinates import (
    SkyCoord,
)
from astropy.constants import c as speed_of_light
from astropy.time import Time
from casacore.tables import makecoldesc, table, taql
from tqdm.auto import tqdm

from jolly_roger.baselines import Baselines, get_baselines_from_ms
from jolly_roger.hour_angles import make_hour_angles_for_ms
from jolly_roger.logging import logger
from jolly_roger.uvws import xyz_to_uvw


def tukey_taper(
    x: np.typing.NDArray[np.floating],
    outer_width: float = np.pi / 4,
    tukey_width: float = np.pi / 8,
) -> np.ndarray:
    x_freq = np.linspace(-np.pi, np.pi, len(x))
    taper = np.ones_like(x_freq)

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
    flags: NDArray[np.bool_]
    uvws: NDArray[np.floating]
    time_centroid: NDArray[np.floating]
    ant_1: NDArray[np.int64]
    ant_2: NDArray[np.int64]
    row_start: int
    chunk_size: int

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
    ant_1: NDArray[np.int64]
    """The first antenna in the baseline."""
    ant_2: NDArray[np.int64]
    """The second antenna in the baseline."""
    row_start: int
    """Starting row index of the data"""
    chunk_size: int
    """Size of the chunked portion of the data"""


def _list_to_array(
    list_of_rows: list[dict[str, Any]],
    key: str
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
    logger.info(f"Length of open table: {table_length} rows")
    
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
            chunk_size=chunk_size
        )
        
        lower_row += chunk_size
        upper_row += chunk_size
    

def get_data_chunks(
    ms_table: table,
    spw_table: table,
    field_table: table,
    chunk_size: int,
    data_column: str,
) -> Generator[DataChunk, None, None]:
    """Yield a collection of rows with appropriate units
    attached to the quantities. These quantities are not
    the same data encoded in the measurement set, e.g. 
    masked array has been formed, astropy units have 
    been attached.

    Args:
        ms_table (table): The main table of the measurement set to iterate over
        spw_table (table): The table describing the spectral windows
        field_table (table): The table describing the fields
        chunk_size (int): The number of rows to return at a time 
        data_column (str): The data column that would be modified

    Yields:
        Generator[DataChunk, None, None]: Representation of the current chunk of rows
    """
    freq_chan = spw_table.getcol("CHAN_FREQ")
    phase_dir = field_table.getcol("PHASE_DIR")

    freq_chan = freq_chan.squeeze() * u.Hz
    target = SkyCoord(*(phase_dir * u.rad).squeeze())
    
    for data_chunk_array in _get_data_chunk_from_main_table(
        ms_table=ms_table, chunk_size=chunk_size, data_column=data_column
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
        ant_1=data_chunk_array.ant_1,
        ant_2=data_chunk_array.ant_2,
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
    ms_tab: table,
    spw_tab: table,
    field_tab: table,
    ant_1: int,
    ant_2: int,
    data_column: str = "DATA",
) -> BaselineData:

    logger.info(f"Getting baseline {ant_1} {ant_2}")

    freq_chan = spw_tab.getcol("CHAN_FREQ")
    phase_dir = field_tab.getcol("PHASE_DIR")

    logger.debug(f"Processing {ant_1=} {ant_2=}")

    baseline_data = _get_baseline_data(
        ms_tab=ms_tab,
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


@dataclass
class DelayTime:
    """Container for delay time and associated metadata."""

    delay_time: np.typing.NDArray[np.complexfloating]
    """ The delay vs time data. shape=(time, delay, pol)"""
    delay: u.Quantity
    """The delay values corresponding to the delay time data."""


def data_to_delay_time(data: BaselineData | DataChunkArray) -> DelayTime:
    logger.info("Converting freq-time to delay-time")
    delay_time = np.fft.fftshift(
        np.fft.fft(data.masked_data.filled(0 + 0j), axis=1), axes=1
    )
    delay = np.fft.fftshift(
        np.fft.fftfreq(
            n=len(data.freq_chan),
            d=np.diff(data.freq_chan).mean(),
        ).decompose()
    )
    return DelayTime(
        delay_time=delay_time,
        delay=delay,
    )


def delay_time_to_data(
    delay_time: DelayTime,
    original_baseline_data: BaselineData,
) -> BaselineData:
    """Convert delay time data back to the original data format."""
    logger.info("Converting delay-time to freq-time")
    new_data = np.fft.ifft(
        np.fft.ifftshift(delay_time.delay_time, axes=1),
        axis=1,
    )
    new_data_masked = np.ma.masked_array(
        new_data,
        mask=original_baseline_data.masked_data.mask,
    )
    new_baseline_data = original_baseline_data
    new_baseline_data.masked_data = new_data_masked
    return new_baseline_data


@dataclass
class DelayRate:
    """Container for delay rate and associated metadata."""

    delay_rate: np.ndarray
    """The delay rate vs time data. shape=(rate, delay, pol)"""
    delay: u.Quantity
    """The delay values corresponding to the delay rate data."""
    rate: u.Quantity
    """The delay rate values corresponding to the delay rate data."""


def data_to_delay_rate(
    baseline_data: BaselineData,
) -> DelayRate:
    """Convert baseline data to delay rate."""

    logger.info("Converting freq-time to delay-rate")
    delay_rate = np.fft.fftshift(np.fft.fft2(baseline_data.masked_data.filled(0 + 0j)))
    delay = np.fft.fftshift(
        np.fft.fftfreq(
            n=len(baseline_data.freq_chan),
            d=np.diff(baseline_data.freq_chan).mean(),
        ).decompose()
    )
    rate = np.fft.fftshift(
        np.fft.fftfreq(
            n=len(baseline_data.time),
            d=np.diff(baseline_data.time.mjd * u.day).mean(),
        ).decompose()
    )

    return DelayRate(
        delay_rate=delay_rate,
        delay=delay,
        rate=rate,
    )


def add_output_column(
    tab: table,
    data_column: str = "DATA",
    output_column: str = "CORRECTED_DATA",
    overwrite: bool = False,
) -> None:

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


def plot_baseline_data(
    baseline_data: BaselineData,
    output_dir: Path,
    suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt
    from astropy.visualization import quantity_support, time_support
    with quantity_support(), time_support():
        data_masked = baseline_data.masked_data
        data_xx = data_masked[..., 0]
        data_yy = data_masked[..., -1]
        data_stokesi = (data_xx + data_yy) / 2
        amp_stokesi = np.abs(data_stokesi)


        fig, ax = plt.subplots()
        im = ax.pcolormesh(
            baseline_data.time,
            baseline_data.freq_chan,
            amp_stokesi.T,
        )
        fig.colorbar(im, ax=ax, label="Stokes I Amplitude / Jy")
        ax.set(
            ylabel=f"Frequency / {baseline_data.freq_chan.unit:latex_inline}",
            title=f"Ant {baseline_data.ant_1} - Ant {baseline_data.ant_2}",
        )
        output_path = output_dir / f"baseline_data_{baseline_data.ant_1}_{baseline_data.ant_2}{suffix}.png"
        fig.savefig(output_path)


def _tukey_tractor(
    data_chunk: DataChunkArray,
    tukey_tractor_options: TukeyTractorOptions,
) -> None:

    delay_time = data_to_delay_time(data=data_chunk)

    taper = tukey_taper(
        x=delay_time.delay,
        outer_width=tukey_tractor_options.outer_width,
        tukey_width=tukey_tractor_options.tukey_width,
    )

    # Delay-time is a 3D array: (time, delay, pol)
    # Taper is 1D: (delay,)
    tapered_delay_time_data = (
        delay_time.delay_time * taper[np.newaxis, :, np.newaxis]
    )
    tapered_delay_time = delay_time
    tapered_delay_time.delay_time = tapered_delay_time_data

    tapered_data = delay_time_to_data(
        delay_time=tapered_delay_time,
        original_baseline_data=data_chunk,
    )
    

    
@dataclass
class TukeyTractorOptions:
    ms_path: Path
    outer_width: float = np.pi / 4
    tukey_width: float = np.pi / 8
    data_column: str = "DATA"
    output_column: str = "CORRECTED_DATA"
    dry_run: bool = False
    make_plots: bool = False
    overwrite: bool = False
    chunk_size: int = 1000


def dumb_tukey_tractor(
    tukey_tractor_options: TukeyTractorOptions,
) -> None:
    baselines = get_baselines_from_ms(tukey_tractor_options.ms_path)
    antennas_for_baselines = baselines.b_map.keys()

    main_table = table(str(tukey_tractor_options.ms_path))
    spw_table = table(str(tukey_tractor_options.ms_path / "SPECTRAL_WINDOW"))
    field_table = table(str(tukey_tractor_options.ms_path / "FIELD"))

    # if not tukey_tractor_options.dry_run:
    #     add_output_column(
    #         tab=ms_tab,
    #         output_column=tukey_tractor_options.output_column,
    #         data_column=tukey_tractor_options.data_column,
    #         overwrite=tukey_tractor_options.overwrite,
    #     )

    for data_chunk in get_data_chunks(
        ms_table=main_table,
        spw_table=spw_table,
        field_table=field_table, 
        chunk_size=tukey_tractor_options.chunk_size,
        data_column=tukey_tractor_options.data_column
    ):
        _tukey_tractor(
                baseline_table=data_chunk,
                tukey_tractor_options=tukey_tractor_options,
            )

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

def plot_tractor_baseline(
        baseline_data: BaselineData,
        delay_time: DelayTime,
        delay_object: u.Quantity,
        delay_phase_center: u.Quantity,
        output_dir: Path,
):
    import matplotlib.pyplot as plt
    from astropy.visualization import quantity_support, time_support
    with quantity_support(), time_support():

        ant_1 = baseline_data.ant_1
        ant_2 = baseline_data.ant_2
        time = baseline_data.time
        delay = delay_time.delay
        delay_time_xx = delay_time.delay_time[..., 0]
        delay_time_yy = delay_time.delay_time[..., -1]
        delay_time_stokesi = (delay_time_xx + delay_time_yy) / 2

        fig, ax = plt.subplots()
        im = ax.pcolormesh(
            time,
            delay,
            np.abs(delay_time_stokesi).T,
            norm=plt.cm.colors.LogNorm()
        )
        ax.set(
            ylabel=f"Delay / {delay.unit:latex_inline}",
            title=f"Baseline {ant_1} - {ant_2}",
        )
        fig.colorbar(im, ax=ax, label="Amplitude")
        ax.plot(time, delay_object, "k--", label="Sun", alpha=1, lw=1)
        ax.plot(
            time,
            delay_phase_center,
            "w-",
            label="phase center",
            alpha=0.5,
        )
        output_path = output_dir / f"tractor_baseline_{ant_1}_{ant_2}.png"
        fig.savefig(output_path)

def tractor_baseline(
    ant_1: int,
    ant_2: int,
    baselines: Baselines,
    uvws_object_baseline: u.Quantity,
    options: TractorOptions,
) -> None:
    baseline_data = get_baseline_data(
        baselines=baselines,
        ant_1=ant_1,
        ant_2=ant_2,
        data_column=options.data_column,
    )

    delay_time = data_to_delay_time(baseline_data=baseline_data)

    _, _, w_coords_object = uvws_object_baseline
    _, _, w_coords_phase_center = baseline_data.uvws_phase_center

    # Subtract 
    delay_object = ((w_coords_object - w_coords_phase_center) / speed_of_light).decompose()
    delay_phase_center = (
        (w_coords_phase_center - w_coords_phase_center) / speed_of_light
    ).decompose()

    if options.make_plots:
        output_dir = options.ms_path.parent / "plots"
        output_dir.mkdir(exist_ok=True)
        plot_baseline_data(
            baseline_data=baseline_data,
            output_dir=output_dir,
            suffix="_original",
        )
        plot_tractor_baseline(
            baseline_data=baseline_data,
            delay_time=delay_time,
            delay_object=delay_object,
            delay_phase_center=delay_phase_center,
            output_dir=output_dir,
        )
        logger.info(f"Plots saved to {output_dir}")
    


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
        )
        dumb_tukey_tractor(tukey_tractor_options=tukey_tractor_options)
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()