from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astropy.visualization import quantity_support, time_support
from casacore.tables import table, taql, makecoldesc
from astropy.coordinates import (
    Angle,
    Longitude,
    SkyCoord,
    EarthLocation,
    AltAz,
    get_sun,
    get_body,
)
from astropy.constants import c as speed_of_light
from tqdm.auto import tqdm

from jolly_roger.baselines import Baselines, get_baselines_from_ms
from jolly_roger.hour_angles import make_hour_angles_for_ms
from jolly_roger.logging import logger
from jolly_roger.uvws import uvw_flagger, xyz_to_uvw


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


def get_baseline_data(
    baselines: Baselines,
    ant_1: int,
    ant_2: int,
    data_column: str = "DATA",
) -> BaselineData:

    ms_path = baselines.ms_path
    logger.info(f"Opening {ms_path=}")
    with (
        table(str(ms_path), ack=False, readonly=True) as ms_tab,
        table(str(ms_path / "SPECTRAL_WINDOW"), ack=False, readonly=True) as spw_tab,
        table(str(ms_path / "FIELD"), ack=False, readonly=True) as field_tab,
    ):
        _ = ms_tab
        freq_chan = spw_tab.getcol("CHAN_FREQ")[:].squeeze() * u.Hz
        phase_dir = field_tab.getcol("PHASE_DIR")[:]
        target = SkyCoord(*(phase_dir * u.rad).squeeze())
        logger.debug(f"Processing {ant_1=} {ant_2=}")
        b_idx = baselines.b_map[(ant_1, ant_2)]
        with taql(
            "select from $ms_tab where ANTENNA1 == $ant_1 and ANTENNA2 == $ant_2",
        ) as subtab:
            data = subtab.getcol(data_column)[:]
            flags = subtab.getcol("FLAG")[:]
            uvws_phase_center = np.swapaxes(
                subtab.getcol("UVW")[:] * u.m, 0, 1
            )
            time = Time(
                subtab.getcol("TIME_CENTROID")[:] * u.s,
                format="mjd",
                scale="utc",
            )
    masked_data = np.ma.masked_array(data, mask=flags)

    return BaselineData(
        masked_data=masked_data,
        freq_chan=freq_chan,
        phase_center=target,
        uvws_phase_center=uvws_phase_center,
        time=time,
    )

@dataclass
class DelayTime:
    """Container for delay time and associated metadata."""

    delay_time: np.typing.NDArray[np.complexfloating]
    """ The delay vs time data. shape=(time, delay, pol)"""
    delay: u.Quantity
    """The delay values corresponding to the delay time data."""

def data_to_delay_time(baseline_data: BaselineData) -> DelayTime:
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
    new_data =  np.fft.ifft(
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
) -> None:
    with table((str(ms_path)), readonly=False) as tab:
        colnames = tab.colnames()
        if output_column in colnames:
            msg = f"Output column {output_column} already exists in the measurement set. Not overwriting."
            raise ValueError(
                msg
            )
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
    update_flags: bool = False
) -> None:
    """Write the output column to the measurement set."""
    with table(str(ms_path), readonly=False) as tab:
        colnames = tab.colnames()
        if output_column not in colnames:
            msg = f"Output column {output_column} does not exist in the measurement set. Cannot write data."
            raise ValueError(msg)

        logger.info(f"Writing {output_column=}")
        tab.putcol(output_column, baseline_data.masked_data.filled(0 + 0j))
        if update_flags:
            # If we want to update the flags, we need to set the flags to False
            # for the output column
            tab.putcol("FLAG", baseline_data.masked_data.mask)
        tab.flush()

def dumb_tukey_tractor(
    ms_path: Path,
    outer_width: float = np.pi / 4,
    tukey_width: float = np.pi / 8,
    data_column: str = "DATA",
    output_column: str = "CORRECTED_DATA",
    dry_run: bool = False,
) -> None:
    baselines = get_baselines_from_ms(ms_path)
    antennas_for_baselines = baselines.b_map.keys()

    if not dry_run:
        add_output_column(
            ms_path=ms_path,
            output_column=output_column,
        )

    for ant_1, ant_2 in tqdm(antennas_for_baselines):
        baseline_data = get_baseline_data(
            baselines=baselines,
            ant_1=ant_1,
            ant_2=ant_2,
            data_column=data_column,
        )

        delay_time = data_to_delay_time(baseline_data=baseline_data)

        taper = tukey_taper(
            x=delay_time.delay,
            outer_width=outer_width,
            tukey_width=tukey_width,
        )

        # Delay-time is a 3D array: (time, delay, pol)
        # Taper is 1D: (delay,)
        tapered_delay_time_data = delay_time.delay_time * taper[np.newaxis, :, np.newaxis]
        tapered_delay_time = delay_time
        tapered_delay_time.delay_time = tapered_delay_time_data

        tapered_baseline_data = delay_time_to_data(
            delay_time=tapered_delay_time,
            original_baseline_data=baseline_data,
        )
        if not dry_run:
            write_output_column(
                ms_path=ms_path,
                output_column=output_column,
                baseline_data=tapered_baseline_data,
            )
        else:
            logger.info(f"Dry run: would write {output_column} for {ant_1=} {ant_2=}")