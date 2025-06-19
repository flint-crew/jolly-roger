"""Utilities and structures around the delay calculations"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from jolly_roger.logging import logger

if TYPE_CHECKING:
    # avoid circular imports
    from jolly_roger.tractor import BaselineData, DataChunk


@dataclass
class DelayTime:
    """Container for delay time and associated metadata."""

    delay_time: NDArray[np.complexfloating]
    """ The delay vs time data. shape=(time, delay, pol)"""
    delay: u.Quantity
    """The delay values corresponding to the delay time data."""


def data_to_delay_time(data: BaselineData | DataChunk) -> DelayTime:
    logger.debug("Converting freq-time to delay-time")
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
    original_data: DataChunk,
) -> DataChunk:
    """Convert delay time data back to the original data format."""
    logger.debug("Converting delay-time to freq-time")
    new_data = np.fft.ifft(
        np.fft.ifftshift(delay_time.delay_time, axes=1),
        axis=1,
    )
    new_data_masked = np.ma.masked_array(
        new_data,
        mask=original_data.masked_data.mask,
    )
    new_data = original_data
    new_data.masked_data = new_data_masked
    return new_data


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
    # This only makes sense when running on time data. Hence
    # asserting the type of BaelineData

    assert isinstance(baseline_data, BaselineData), (
        f"baseline_data is type={type(baseline_data)}, but needs to be BaselineData"
    )

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
