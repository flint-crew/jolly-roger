from __future__ import annotations

from argparse import ArgumentParser
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    SkyCoord,
)
from astropy.time import Time
from casacore.tables import makecoldesc, table, taql
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from jolly_roger.baselines import Baselines, get_baselines_from_ms
from jolly_roger.logging import logger


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

def get_baseline_data(
    baselines: Baselines,
    ant_1: int,
    ant_2: int,
    data_column: str = "DATA",
) -> BaselineData:
    ms_path = baselines.ms_path
    logger.info(f"Opening {ms_path=} - getting baseline {ant_1} {ant_2}")
    with (
        table(str(ms_path), ack=False, readonly=True) as ms_tab,
        table(str(ms_path / "SPECTRAL_WINDOW"), ack=False, readonly=True) as spw_tab,
        table(str(ms_path / "FIELD"), ack=False, readonly=True) as field_tab,
    ):
        _ = ms_tab
        freq_chan = spw_tab.getcol("CHAN_FREQ")
        phase_dir = field_tab.getcol("PHASE_DIR")

        logger.debug(f"Processing {ant_1=} {ant_2=}")
        with taql(
            "select from $ms_tab where ANTENNA1 == $ant_1 and ANTENNA2 == $ant_2",
        ) as subtab:
            logger.info(f"Opening subtable for baseline {ant_1} {ant_2}")
            data = subtab.getcol(data_column)
            flags = subtab.getcol("FLAG")
            uvws = subtab.getcol("UVW")
            time_centroid = subtab.getcol("TIME_CENTROID")

        freq_chan = freq_chan.squeeze() * u.Hz
        target = SkyCoord(*(phase_dir * u.rad).squeeze())
        uvws_phase_center = np.swapaxes(uvws * u.m, 0, 1)
        time = Time(
            time_centroid.squeeze() * u.s,
            format="mjd",
            scale="utc",
        )
    masked_data = np.ma.masked_array(data, mask=flags)

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


def data_to_delay_time(baseline_data: BaselineData) -> DelayTime:
    logger.info("Converting freq-time to delay-time")
    delay_time = np.fft.fftshift(
        np.fft.fft(baseline_data.masked_data.filled(0 + 0j), axis=1), axes=1
    )
    delay = np.fft.fftshift(
        np.fft.fftfreq(
            n=len(baseline_data.freq_chan),
            d=np.diff(baseline_data.freq_chan).mean(),
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
    ms_path: Path,
    data_column: str = "DATA",
    output_column: str = "CORRECTED_DATA",
    overwrite: bool = False,
) -> None:
    with table((str(ms_path)), readonly=False) as tab:
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
        output_path = output_dir / f"baseline_data_{baseline_data.ant_1}_{baseline_data.ant_2}.png"
        fig.savefig(output_path)


async def _tukey_tractor_baseline(
    ant_1: int,
    ant_2: int,
    baselines: Baselines,
    tukey_tractor_options: TukeyTractorOptions,
) -> None:
    baseline_data = await asyncio.to_thread(
        get_baseline_data,
        baselines=baselines,
        ant_1=ant_1,
        ant_2=ant_2,
        data_column=tukey_tractor_options.data_column,
    )
    logger.info(f"{baseline_data.masked_data.shape=}")
    return

    delay_time = data_to_delay_time(baseline_data=baseline_data)

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

    tapered_baseline_data = delay_time_to_data(
        delay_time=tapered_delay_time,
        original_baseline_data=baseline_data,
    )
    if not tukey_tractor_options.dry_run:
        await asyncio.to_thread(
            write_output_column,
            ms_path=tukey_tractor_options.ms_path,
            output_column=tukey_tractor_options.output_column,
            baseline_data=tapered_baseline_data,
        )
    else:
        logger.info(
            f"Dry run: would write {tukey_tractor_options.output_column} for baseline {ant_1} {ant_2}"
        )

    if tukey_tractor_options.make_plots:
        output_dir = tukey_tractor_options.ms_path.parent / "plots"
        output_dir.mkdir(exist_ok=True)
        plot_baseline_data(
            baseline_data=tapered_baseline_data,
            output_dir=output_dir,
        )
        logger.info(f"Plots saved to {output_dir}")

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


semaphore = asyncio.Semaphore(4)  # Start with 4; tune higher if I/O allows

async def _limited_tukey_tractor_baseline(*args, **kwargs):
    async with semaphore:
        return await _tukey_tractor_baseline(*args, **kwargs)

async def dumb_tukey_tractor(
    tukey_tractor_options: TukeyTractorOptions,
) -> None:
    baselines = get_baselines_from_ms(tukey_tractor_options.ms_path)
    antennas_for_baselines = baselines.b_map.keys()

    if not tukey_tractor_options.dry_run:
        add_output_column(
            ms_path=tukey_tractor_options.ms_path,
            output_column=tukey_tractor_options.output_column,
            data_column=tukey_tractor_options.data_column,
            overwrite=tukey_tractor_options.overwrite,
        )

    coros = []
    for ant_1, ant_2 in antennas_for_baselines:
        coro = _limited_tukey_tractor_baseline(
            ant_1=ant_1,
            ant_2=ant_2,
            baselines=baselines,
            tukey_tractor_options=tukey_tractor_options,
        )
        # coros.append(coro)
        # task = asyncio.create_task(coro)
        coros.append(coro)
    await tqdm.gather(*coros)

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
        asyncio.run(dumb_tukey_tractor(tukey_tractor_options=tukey_tractor_options))
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()