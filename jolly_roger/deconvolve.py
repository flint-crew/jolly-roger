"""Remove bright objects outside the field by CLEAN in delay space.

Each visibility spectrum is transformed to delay with an FFT. Its delay spread
function (the response to a point source) is the FFT of the channel weights, so
it is known exactly, flags included. The dirty delay spectrum is deconvolved
with a Hogbom CLEAN (``rm_lite.utils.clean.minor_loop``) over the whole delay
axis, using a mask that grows from the brightest emission and a threshold set
by the noise. Nothing about the size, structure or spectrum of the object is
assumed. Unless it is given, the noise is estimated from each spectrum (see
``estimate_visibility_noise``). Only the components outside the field's delay range are subtracted
from the visibilities; the field is cleaned at the same time but left in place.

``rm-lite`` is an optional dependency, installed with ``jolly-roger[clean]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jolly_roger.logging import logger
from jolly_roger.tractor import DataChunk, TaperedChunkResult


class MissingCleanDependencyError(ImportError):
    pass


def _import_minor_loop() -> Any:
    """Import rm-lite on first use, so it is only needed for delay CLEAN"""
    try:
        from rm_lite.utils import clean  # noqa: PLC0415
        from rm_lite.utils.logging import quiet_logs  # noqa: PLC0415
    except ImportError as error:
        msg = "Delay CLEAN needs rm-lite. Install it with: pip install 'jolly-roger[clean]'"
        raise MissingCleanDependencyError(msg) from error
    return clean, quiet_logs


@dataclass(frozen=True)
class DelayCleanOptions:
    """Settings for CLEAN in delay space"""

    oversample: int = 4
    """Factor by which the delay axis is sampled more finely than 1 / bandwidth"""
    gain: float = 0.1
    """Fraction of the residual peak removed in each CLEAN iteration"""
    mask_sigma: float = 5.0
    """The CLEAN mask grows into delays brighter than this many times the delay noise"""
    threshold_sigma: float = 3.0
    """CLEAN stops when the residual inside the mask falls below this many times the delay noise"""
    max_iter: int = 100_000
    """Largest number of CLEAN iterations per spectrum"""
    field_edge_resolutions: float = 2.0
    """Width, in units of 1 / bandwidth, of the band just outside the field's delay range that is checked for subtracted emission"""
    field_edge_fraction: float = 0.01
    """The object reaches the field when the emission subtracted in that band exceeds this fraction of the brightest emission subtracted"""


@dataclass(frozen=True)
class DelayCleanResult:
    """Outcome of CLEAN on one visibility spectrum"""

    model: NDArray[np.complexfloating]
    """The emission outside the field's delay range, in each channel, to subtract from the visibilities"""
    iterations: int
    """Number of CLEAN iterations"""
    delay_noise: float
    """Noise in the delay spectrum (per real or imaginary part) used to set the mask and threshold"""
    reaches_field: bool
    """Whether subtracted emission lies just outside the field's delay range, so the object and field may overlap"""


def delay_axis(
    number_of_channels: int, channel_width_hz: float, oversample: int
) -> NDArray[np.floating]:
    """Delays, in seconds, of the oversampled delay spectrum in increasing order

    Args:
        number_of_channels (int): Number of channels in the spectrum
        channel_width_hz (float): Channel width, in Hz
        oversample (int): Oversampling factor of the delay axis

    Returns:
        NDArray[np.floating]: The delay of each sample, in seconds
    """
    size = oversample * number_of_channels
    return np.fft.fftshift(np.fft.fftfreq(size, d=abs(channel_width_hz)))


def _blackman_harris(size: int) -> NDArray[np.floating]:
    """Four-term Blackman-Harris window, whose sidelobes are 92 dB down"""
    phase = 2 * np.pi * np.arange(size) / max(size - 1, 1)
    return (
        0.35875
        - 0.48829 * np.cos(phase)
        + 0.14128 * np.cos(2 * phase)
        - 0.01168 * np.cos(3 * phase)
    )


def _unflagged_segments(used: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Start and end (exclusive) of each run of consecutive used channels"""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], used.astype(int), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def estimate_visibility_noise(
    values: NDArray[np.complexfloating],
    weights: NDArray[np.floating],
    min_segment_channels: int = 32,
    quantile: float = 0.25,
) -> float | None:
    """Estimate the noise of one visibility, per real or imaginary part, from
    the spectrum itself.

    Noise is spread evenly over all delays, while emission occupies a limited
    range of them. Each run of at least ``min_segment_channels`` unflagged
    channels is windowed (so that bright emission has no sidelobes across the
    delay axis, from either the band edges or the flags) and transformed to
    delay. The amplitudes of noise follow a Rayleigh distribution, and emission
    only adds to them, so a low quantile of all amplitudes gives the noise even
    when emission is above the noise over most of the delay axis. Any emission
    that remains biases the estimate high, never low.

    Args:
        values (NDArray[np.complexfloating]): Visibilities of regularly spaced channels
        weights (NDArray[np.floating]): Weight of each channel. Channels with zero weight are not used.
        min_segment_channels (int, optional): Shortest run of used channels to include. Defaults to 32.
        quantile (float, optional): Quantile of the delay amplitudes to use. Defaults to 0.25.

    Returns:
        float | None: The noise, or None if no run of channels is long enough
    """
    amplitudes = []
    for start, end in _unflagged_segments(np.asarray(weights) > 0):
        if end - start < min_segment_channels:
            continue
        window = _blackman_harris(end - start)
        spectrum = np.fft.fft(values[start:end] * window)
        amplitudes.append(np.abs(spectrum) / np.sqrt(np.sum(window**2)))
    if not amplitudes:
        return None
    # For a Rayleigh distribution the quantile q is sigma * sqrt(-2 ln(1 - q))
    rayleigh_quantile = np.sqrt(-2 * np.log(1 - quantile))
    return float(np.quantile(np.concatenate(amplitudes), quantile) / rayleigh_quantile)


def _reaches_field(
    model: NDArray[np.complexfloating],
    delays_s: NDArray[np.floating],
    field_delay_s: float,
    edge_s: float,
    oversample: int,
    floor: float,
    fraction: float,
) -> bool:
    """Whether the emission subtracted outside the field runs into the field.

    CLEAN components are summed over one resolution element. The object is
    taken to reach the field when, just outside the field's delay range, this
    is both above ``floor`` and above ``fraction`` of its largest value outside
    the field. The second condition keeps the faint wings of the field itself
    (e.g. from its spectral index) from counting as the object.
    """
    per_resolution = np.convolve(np.abs(model), np.ones(oversample), mode="same")
    outside_field = np.abs(delays_s) > field_delay_s
    if not np.any(per_resolution[outside_field] > 0):
        return False
    edge = outside_field & (np.abs(delays_s) <= field_delay_s + edge_s)
    peak_outside = float(np.max(per_resolution[outside_field]))
    edge_peak = float(np.max(per_resolution[edge], initial=0.0))
    return edge_peak > max(floor, fraction * peak_outside)


def _model_to_values(
    model: NDArray[np.complexfloating], number_of_channels: int
) -> NDArray[np.complexfloating]:
    """Visibilities in each channel of CLEAN components on the oversampled delay axis"""
    size = len(model)
    return size * np.fft.ifft(np.fft.ifftshift(model))[:number_of_channels]


def delay_clean_spectrum(
    values: NDArray[np.complexfloating],
    weights: NDArray[np.floating],
    channel_width_hz: float,
    field_delay_s: float,
    options: DelayCleanOptions | None = None,
    visibility_noise: float | None = None,
) -> DelayCleanResult:
    """CLEAN one visibility spectrum in delay, and return the emission outside
    the field's delay range.

    Args:
        values (NDArray[np.complexfloating]): Visibilities of regularly spaced channels
        weights (NDArray[np.floating]): Weight of each channel. Zero for flagged or excluded channels.
        channel_width_hz (float): Channel width, in Hz
        field_delay_s (float): Half-width of the field's delay range, in seconds. Emission within it is kept.
        options (DelayCleanOptions | None, optional): CLEAN settings. Defaults to None (the defaults).
        visibility_noise (float | None, optional): Noise of one visibility, per real or imaginary part, at unit weight. If None it is estimated with ``estimate_visibility_noise``. Defaults to None.

    Returns:
        DelayCleanResult: The model to subtract and a summary of the CLEAN. The model is zero if the noise could not be estimated.
    """
    clean, _ = _import_minor_loop()
    options = options or DelayCleanOptions()

    number_of_channels = len(values)
    weights = np.asarray(weights, dtype=float)
    weight_sum = weights.sum()
    if visibility_noise is None and weight_sum > 0:
        visibility_noise = estimate_visibility_noise(values, weights)
    if weight_sum <= 0 or visibility_noise is None:
        return DelayCleanResult(
            model=np.zeros(number_of_channels, dtype=complex),
            iterations=0,
            delay_noise=0.0,
            reaches_field=False,
        )

    size = options.oversample * number_of_channels
    delays_s = delay_axis(number_of_channels, channel_width_hz, options.oversample)
    weighted = np.where(weights > 0, values, 0) * weights

    dirty = np.fft.fftshift(np.fft.fft(weighted, n=size)) / weight_sum
    delay_spread_function = np.fft.fftshift(np.fft.fft(weights, n=size)) / weight_sum
    # The delay axis is cyclic, so three periods give the shifted delay spread
    # function rm-lite expects for every peak position
    delay_spread_function = np.tile(delay_spread_function, 3)
    index = np.arange(size, dtype=float)
    index_double = np.arange(3 * size, dtype=float) - size

    # Noise per delay sample, from the noise per visibility at unit weight
    delay_noise = visibility_noise * np.sqrt(np.sum(weights**2)) / weight_sum

    result = clean.minor_loop(
        clean.MinorLoopArrays(
            resid_fdf_spectrum_mask=np.ma.masked_array(
                dirty, mask=np.zeros(size, dtype=bool)
            ),
            phi_arr_radm2=index,
            phi_double_arr_radm2=index_double,
            rmsf_spectrum=delay_spread_function,
            rmsf_fwhm=float(options.oversample),
        ),
        clean.MinorLoopOptions(
            max_iter=options.max_iter,
            gain=options.gain,
            mask_threshold=options.mask_sigma * delay_noise,
            stopping_threshold=options.threshold_sigma * delay_noise,
            update_mask=True,
            noise=delay_noise,
        ),
    )
    model = result.model_fdf_spectrum

    outside_field = np.abs(delays_s) > field_delay_s
    reaches_field = _reaches_field(
        model=model,
        delays_s=delays_s,
        field_delay_s=field_delay_s,
        edge_s=options.field_edge_resolutions
        / (number_of_channels * abs(channel_width_hz)),
        oversample=options.oversample,
        floor=options.mask_sigma * delay_noise,
        fraction=options.field_edge_fraction,
    )

    model_values = _model_to_values(
        np.where(outside_field, model, 0), number_of_channels
    )

    return DelayCleanResult(
        model=model_values,
        iterations=result.iter_count,
        delay_noise=delay_noise,
        reaches_field=reaches_field,
    )


def compute_delay_clean(
    data_chunk: DataChunk,
    field_delay_s: float | NDArray[np.floating],
    options: DelayCleanOptions | None = None,
    visibility_noise: float | None = None,
) -> TaperedChunkResult:
    """Remove emission outside the field's delay range from every row and
    polarisation of a chunk with CLEAN in delay. Rows where the subtracted
    emission reaches the field are flagged.

    Args:
        data_chunk (DataChunk): The visibilities, shaped (row, channel, polarisation). Masked values are given zero weight.
        field_delay_s (float | NDArray[np.floating]): Half-width of the field's delay range, in seconds, for all rows or for each row
        options (DelayCleanOptions | None, optional): CLEAN settings. Defaults to None (the defaults).
        visibility_noise (float | None, optional): Noise of one visibility, per real or imaginary part. If None it is estimated from each spectrum. Defaults to None.

    Returns:
        TaperedChunkResult: The cleaned visibilities and any new flags
    """
    _, quiet_logs = _import_minor_loop()
    options = options or DelayCleanOptions()

    freq_hz = data_chunk.freq_chan.to("Hz").value
    channel_width_hz = float(np.mean(np.diff(freq_hz)))
    data = np.array(data_chunk.masked_data.data, copy=True)
    flags = np.ma.getmaskarray(data_chunk.masked_data)
    number_of_rows, _, number_of_pols = data.shape
    field_delays = np.broadcast_to(
        np.asarray(field_delay_s, dtype=float), (number_of_rows,)
    )

    rows_reaching_field = np.zeros(number_of_rows, dtype=bool)
    with quiet_logs(logging.WARNING):
        for row in range(number_of_rows):
            for pol in range(number_of_pols):
                weights = (~flags[row, :, pol]).astype(float)
                if not weights.any():
                    continue
                result = delay_clean_spectrum(
                    values=data[row, :, pol],
                    weights=weights,
                    channel_width_hz=channel_width_hz,
                    field_delay_s=float(field_delays[row]),
                    options=options,
                    visibility_noise=visibility_noise,
                )
                data[row, :, pol] -= result.model
                rows_reaching_field[row] |= result.reaches_field

    if rows_reaching_field.any():
        logger.info(
            f"Flagging {rows_reaching_field.sum()} of {number_of_rows} rows where the object reaches the field in delay"
        )
    new_flags = flags | rows_reaching_field[:, None, None]

    return TaperedChunkResult(
        chunk_size=data_chunk.chunk_size,
        nothing_to_do=False,
        update_data=True,
        data_chunk=replace(
            data_chunk, masked_data=np.ma.masked_array(data, mask=new_flags)
        ),
        update_flags=bool(rows_reaching_field.any()),
        flags=new_flags,
    )
