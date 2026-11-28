"""Remove bright objects outside the field by CLEAN in delay space.

Each visibility spectrum is transformed to delay with an FFT. Its delay spread
function (the response to a point source) is the FFT of the channel weights, so
it is known exactly, flags included. The dirty delay spectrum is deconvolved
with a Hogbom CLEAN (``rm_lite.utils.clean.minor_loop_batch``) over the whole
delay axis, using a mask that grows from the brightest emission and a threshold
set by the noise. Nothing about the size, structure or spectrum of the object is
assumed. Unless it is given, the noise is estimated from each spectrum (see
``estimate_visibility_noise``). Only the components outside the field's delay range are subtracted
from the visibilities; the field is cleaned at the same time but left in place.

``rm-lite`` is an optional dependency, installed with ``jolly-roger[clean]``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal, stats

from jolly_roger.baselines import OpenMSTables
from jolly_roger.history import add_history, dataclass_to_app_params
from jolly_roger.logging import logger
from jolly_roger.rates import (
    RateFilterRun,
    RateFilterSettings,
    SegmentFilterResult,
    SegmentGeometry,
    SegmentOutcome,
    describe_rate_filter,
    make_rate_filter_run,
)
from jolly_roger.tractor import (
    DataChunk,
    RateFilterOptions,
    TaperedChunkResult,
    TractorOptions,
    _get_baseline_time_indicies,
    get_plot_baselines,
    make_plot_results,
    make_rate_filter_payload,
    prepare_tractor,
    read_baseline_flags,
    run_tractor_chunks,
)
from jolly_roger.utils import log_dataclass_attributes, log_jolly_roger_version
from jolly_roger.uvws import WDelays
from jolly_roger.wrap import calculate_nyquist_zone


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
    if not hasattr(clean, "minor_loop_batch"):
        msg = "Delay CLEAN needs rm-lite>=2026.11.0. Upgrade it with: pip install -U 'jolly-roger[clean]'"
        raise MissingCleanDependencyError(msg)
    return clean, quiet_logs


@dataclass(frozen=True)
class DelayCleanOptions:
    """Settings for CLEAN in delay space"""

    oversample: int = 4
    """Factor by which the delay axis is sampled more finely than 1 / bandwidth"""
    gain: float = 0.3
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
    edge_residual: float | None = None
    """Residual after CLEAN in the channels at each end of the band, as a multiple of the noise"""
    mid_residual: float | None = None
    """Residual after CLEAN in the remaining channels, as a multiple of the noise"""


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
    noise = _estimate_visibility_noises(
        values=np.asarray(values)[np.newaxis],
        weights=weights,
        min_segment_channels=min_segment_channels,
        quantile=quantile,
    )[0]
    return None if np.isnan(noise) else float(noise)


def _estimate_visibility_noises(
    values: NDArray[np.complexfloating],
    weights: NDArray[np.floating],
    min_segment_channels: int = 32,
    quantile: float = 0.25,
) -> NDArray[np.floating]:
    """``estimate_visibility_noise`` for each row of ``values``, which share ``weights``. NaN where it cannot be estimated."""
    runs, _ = ndimage.label(np.asarray(weights) > 0)
    amplitudes = []
    for run in ndimage.find_objects(runs):
        segment = values[:, run[0]]
        if segment.shape[1] < min_segment_channels:
            continue
        # Four-term Blackman-Harris window, whose sidelobes are 92 dB down
        # (Harris 1978, Proc. IEEE 66, 51), so that bright emission does not
        # spread across the delay axis from the ends of the run
        window = signal.windows.blackmanharris(segment.shape[1])
        spectrum = np.fft.fft(segment * window, axis=-1)
        amplitudes.append(np.abs(spectrum) / np.sqrt(np.sum(window**2)))
    if not amplitudes:
        return np.full(len(values), np.nan)
    # The amplitude of complex Gaussian noise with sigma per real or imaginary
    # part follows a Rayleigh distribution with scale sigma
    return np.asarray(
        np.quantile(np.concatenate(amplitudes, axis=-1), quantile, axis=-1)
        / stats.rayleigh.ppf(quantile),
        dtype=float,
    )


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


EDGE_CHANNELS = 8
"""Number of used channels at each end of the band whose residual is reported separately"""


def _residual_by_band_position(
    residual: NDArray[np.complexfloating],
    used: NDArray[np.bool_],
    noise: float,
) -> tuple[float | None, float | None]:
    """RMS of the residual after CLEAN at the band edges and mid-band, as a
    multiple of the noise. A cut-off at the band edges leaves emission there
    that CLEAN does not remove, so the edges are reported apart."""
    used_channels = np.flatnonzero(used)
    if len(used_channels) < 4 * EDGE_CHANNELS or noise <= 0:
        return None, None
    edge = np.r_[used_channels[:EDGE_CHANNELS], used_channels[-EDGE_CHANNELS:]]
    mid = used_channels[EDGE_CHANNELS:-EDGE_CHANNELS]
    # The noise is per real or imaginary part, so complex noise has rms noise * sqrt(2)
    noise_rms = noise * np.sqrt(2)
    return (
        float(np.sqrt(np.mean(np.abs(residual[edge]) ** 2)) / noise_rms),
        float(np.sqrt(np.mean(np.abs(residual[mid]) ** 2)) / noise_rms),
    )


def _model_to_values(
    model: NDArray[np.complexfloating], number_of_channels: int
) -> NDArray[np.complexfloating]:
    """Visibilities in each channel of CLEAN components on the oversampled delay
    axis, the last axis of ``model``"""
    size = model.shape[-1]
    return (
        size
        * np.fft.ifft(np.fft.ifftshift(model, axes=-1), axis=-1)[
            ..., :number_of_channels
        ]
    )


def _no_clean_result(number_of_channels: int) -> DelayCleanResult:
    """The result for a spectrum that is not cleaned: no weight or no noise"""
    return DelayCleanResult(
        model=np.zeros(number_of_channels, dtype=complex),
        iterations=0,
        delay_noise=0.0,
        reaches_field=False,
    )


def _clean_batch(
    values: NDArray[np.complexfloating],
    weights: NDArray[np.floating],
    noise: NDArray[np.floating],
    options: DelayCleanOptions,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating], NDArray[np.int_]]:
    """CLEAN a stack of regularly sampled series that share their weights,
    over the conjugate axis oversampled by ``options.oversample`` (as ``delay_axis``)

    Args:
        values (NDArray[np.complexfloating]): The series, shaped (series, sample)
        weights (NDArray[np.floating]): Weight of each sample, shared by every series, with a positive sum
        noise (NDArray[np.floating]): Noise of one sample of each series, per real or imaginary part, at unit weight
        options (DelayCleanOptions): CLEAN settings

    Returns:
        tuple[NDArray[np.complexfloating], NDArray[np.floating], NDArray[np.int_]]: The CLEAN components of each series on the conjugate axis, the noise on that axis, and the iterations of each
    """
    clean, _ = _import_minor_loop()
    size = options.oversample * values.shape[-1]
    weight_sum = weights.sum()
    weighted = np.where(weights > 0, values, 0) * weights

    dirty = np.fft.fftshift(np.fft.fft(weighted, n=size, axis=-1), axes=-1) / weight_sum
    spread_function = np.fft.fftshift(np.fft.fft(weights, n=size)) / weight_sum
    # The axis is cyclic, so three periods give the shifted spread function
    # rm-lite expects for every peak position
    spread_function = np.tile(spread_function, 3)
    index = np.arange(size, dtype=float)
    index_double = np.arange(3 * size, dtype=float) - size

    # Noise per sample of the conjugate axis, from the noise per sample at unit weight
    conjugate_noise = noise * np.sqrt(np.sum(weights**2)) / weight_sum

    batch = clean.minor_loop_batch(
        clean.MinorLoopBatchArrays(
            resid_fdf_arr_mask=np.ma.masked_array(
                dirty, mask=np.zeros(dirty.shape, dtype=bool)
            ),
            phi_arr_radm2=index,
            phi_double_arr_radm2=index_double,
            rmsf_arr=spread_function,
        ),
        clean.MinorLoopBatchOptions(
            max_iter=options.max_iter,
            gain=options.gain,
            mask_threshold=options.mask_sigma * conjugate_noise,
            stopping_threshold=options.threshold_sigma * conjugate_noise,
            update_mask=True,
            noise=conjugate_noise,
            # Chunks already run on the tractor's threads, and numba's
            # workqueue layer aborts when launched from several threads
            parallel=False,
        ),
    )
    return batch.model_fdf_arr, conjugate_noise, batch.iter_count_arr


@dataclass
class SelectedClean:
    """The result of ``clean_and_select``"""

    model: NDArray[np.complexfloating]
    """Model of the kept components at every sample, shaped as the series"""
    components: NDArray[np.complexfloating]
    """All CLEAN components on the conjugate axis, shape (series, axis)"""
    threshold: NDArray[np.floating]
    """CLEAN's stopping threshold on the conjugate axis for each series"""
    cleaned: NDArray[np.bool_]
    """Series that were CLEANed and converged"""


def clean_and_select(
    values: NDArray[np.complexfloating],
    weights: NDArray[np.floating],
    keep: NDArray[np.bool_],
    options: DelayCleanOptions | None = None,
    visibility_noise: float | NDArray[np.floating] | None = None,
) -> SelectedClean:
    """CLEAN regularly sampled series that share their weights, and model the
    components in ``keep`` at every sample

    Args:
        values (NDArray[np.complexfloating]): The series, shaped (series, sample)
        weights (NDArray[np.floating]): Weight of each sample, shared by every series
        keep (NDArray[np.bool_]): Components to keep on the conjugate axis given by ``delay_axis``, for all series or shaped (series, axis)
        options (DelayCleanOptions | None, optional): CLEAN settings. Defaults to None (the defaults).
        visibility_noise (float | NDArray[np.floating] | None, optional): Noise of one sample, per real or imaginary part, at unit weight, for all series or for each. If None it is estimated with ``estimate_visibility_noise``. Defaults to None.

    Returns:
        SelectedClean: The model of the kept components, all components, the threshold, and which series were CLEANed
    """
    options = options or DelayCleanOptions()
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    number_of_series, number_of_samples = values.shape
    size = options.oversample * number_of_samples
    result = SelectedClean(
        model=np.zeros_like(values, dtype=complex),
        components=np.zeros((number_of_series, size), dtype=complex),
        threshold=np.full(number_of_series, np.inf),
        cleaned=np.zeros(number_of_series, dtype=bool),
    )
    if weights.sum() <= 0:
        return result
    noise = (
        _estimate_visibility_noises(values, weights)
        if visibility_noise is None
        else np.broadcast_to(
            np.asarray(visibility_noise, dtype=float), (number_of_series,)
        )
    )
    to_clean = np.flatnonzero(~np.isnan(noise))
    if to_clean.size == 0:
        return result
    components, conjugate_noise, iterations = _clean_batch(
        values=values[to_clean], weights=weights, noise=noise[to_clean], options=options
    )
    keep = np.broadcast_to(keep, (number_of_series, size))[to_clean]
    result.model[to_clean] = _model_to_values(
        np.where(keep, components, 0), number_of_samples
    )
    result.components[to_clean] = components
    result.threshold[to_clean] = options.threshold_sigma * conjugate_noise
    result.cleaned[to_clean] = iterations < options.max_iter
    return result


def delay_clean_spectra(
    values: NDArray[np.complexfloating],
    weights: NDArray[np.floating],
    channel_width_hz: float,
    field_delay_s: float | NDArray[np.floating],
    options: DelayCleanOptions | None = None,
    visibility_noise: float | NDArray[np.floating] | None = None,
) -> list[DelayCleanResult]:
    """CLEAN spectra that share their channel weights in one batch, as ``delay_clean_spectrum`` does each.

    Args:
        values (NDArray[np.complexfloating]): Visibilities, shaped (spectrum, channel), of regularly spaced channels
        weights (NDArray[np.floating]): Weight of each channel, shared by every spectrum. Zero for flagged or excluded channels.
        channel_width_hz (float): Channel width, in Hz
        field_delay_s (float | NDArray[np.floating]): Half-width of the field's delay range, in seconds, for all spectra or for each. Emission within it is kept.
        options (DelayCleanOptions | None, optional): CLEAN settings. Defaults to None (the defaults).
        visibility_noise (float | NDArray[np.floating] | None, optional): Noise of one visibility, per real or imaginary part, at unit weight, for all spectra or for each. NaN means the noise is not known, so the spectrum is not cleaned. If None it is estimated from each spectrum with ``estimate_visibility_noise``. Defaults to None.

    Returns:
        list[DelayCleanResult]: The model to subtract and a summary of the CLEAN, for each spectrum. The model is zero where the noise could not be estimated.
    """
    _import_minor_loop()
    options = options or DelayCleanOptions()

    values = np.asarray(values)
    if values.ndim != 2:
        msg = f"values must be shaped (spectrum, channel), got {values.shape}"
        raise ValueError(msg)
    number_of_spectra, number_of_channels = values.shape
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (number_of_channels,):
        msg = f"weights must be shaped ({number_of_channels},), got {weights.shape}"
        raise ValueError(msg)
    field_delays = np.broadcast_to(
        np.asarray(field_delay_s, dtype=float), (number_of_spectra,)
    )

    weight_sum = weights.sum()
    if weight_sum <= 0:
        return [_no_clean_result(number_of_channels) for _ in range(number_of_spectra)]

    noise = (
        _estimate_visibility_noises(values, weights)
        if visibility_noise is None
        else np.broadcast_to(
            np.asarray(visibility_noise, dtype=float), (number_of_spectra,)
        )
    )
    to_clean = np.flatnonzero(~np.isnan(noise))

    results = [_no_clean_result(number_of_channels) for _ in range(number_of_spectra)]
    if to_clean.size == 0:
        return results
    values_to_clean = values[to_clean]
    noise = noise[to_clean]

    delays_s = delay_axis(number_of_channels, channel_width_hz, options.oversample)
    models, delay_noise, iterations = _clean_batch(
        values=values_to_clean, weights=weights, noise=noise, options=options
    )

    outside_field = np.abs(delays_s) > field_delays[to_clean, None]
    model_values = _model_to_values(
        np.where(outside_field, models, 0), number_of_channels
    )
    residuals = values_to_clean - _model_to_values(models, number_of_channels)
    edge_s = options.field_edge_resolutions / (
        number_of_channels * abs(channel_width_hz)
    )

    for position, spectrum in enumerate(to_clean):
        reaches_field = _reaches_field(
            model=models[position],
            delays_s=delays_s,
            field_delay_s=float(field_delays[spectrum]),
            edge_s=edge_s,
            oversample=options.oversample,
            floor=options.mask_sigma * delay_noise[position],
            fraction=options.field_edge_fraction,
        )
        edge_residual, mid_residual = _residual_by_band_position(
            residual=residuals[position],
            used=weights > 0,
            noise=float(noise[position]),
        )
        results[spectrum] = DelayCleanResult(
            model=model_values[position],
            iterations=int(iterations[position]),
            delay_noise=float(delay_noise[position]),
            reaches_field=reaches_field,
            edge_residual=edge_residual,
            mid_residual=mid_residual,
        )
    return results


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
    return delay_clean_spectra(
        values=np.asarray(values)[np.newaxis],
        weights=weights,
        channel_width_hz=channel_width_hz,
        field_delay_s=field_delay_s,
        options=options,
        visibility_noise=visibility_noise,
    )[0]


def _object_rate_bands(
    geometry: SegmentGeometry,
    freq_hz: NDArray[np.floating],
    duration_s: float,
    rate_margin_lobes: float,
    min_margin_lobes: float,
) -> tuple[list[tuple[NDArray[np.floating], NDArray[np.floating]]], bool]:
    """Each active object's fringe-rate band in each channel, widened by up to
    ``rate_margin_lobes`` / T but never into the field's range, and whether
    every object is at least ``min_margin_lobes`` / T from the field"""
    frequency = np.abs(freq_hz)
    field_rate_hz = float(np.max(geometry.field_rate)) * frequency + 1 / duration_s
    bands = []
    for index in range(len(geometry.object_names)):
        active = geometry.object_active[index]
        if not active.any():
            continue
        rate = geometry.object_rate[index][active]
        low, high = rate.min() * frequency, rate.max() * frequency
        gap = np.where(low > 0, low - field_rate_hz, -field_rate_hz - high)
        if np.any(gap < min_margin_lobes / duration_s):
            return [], False
        margin = np.minimum(rate_margin_lobes / duration_s, gap)
        bands.append((low - margin, high + margin))
    return bands, True


def rate_clean_segment(
    data: NDArray[np.complexfloating],
    flags: NDArray[np.bool_],
    is_pad: NDArray[np.bool_],
    time_s: NDArray[np.floating],
    freq_hz: NDArray[np.floating],
    geometry: SegmentGeometry,
    options: DelayCleanOptions | None = None,
    visibility_noise: float | None = None,
    rate_margin_lobes: float = 8.0,
    min_margin_lobes: float = 4.0,
    spill_lobes: float = 4.0,
) -> SegmentFilterResult:
    """Remove the objects from a baseline segment by CLEAN in fringe rate.

    Each channel and polarisation is CLEANed along time, so the field and the
    objects are separated by fringe rate alone and the objects' delay sidelobes
    never arise. Only components in each object's fringe-rate band are
    subtracted, evaluated at every row; the residual and everything else are
    kept. A row is not recovered if CLEAN did not run or converge, or if
    components just outside an object's band (the object varying faster than the
    margin) exceed CLEAN's threshold and ``field_edge_fraction`` of the object
    subtracted at that row.

    Args:
        data (NDArray[np.complexfloating]): Visibilities, shape (row, chan, pol)
        flags (NDArray[np.bool_]): Flags, shape (row, chan, pol)
        is_pad (NDArray[np.bool_]): Whether each row is a pad row, never recovered
        time_s (NDArray[np.floating]): Time of each row, in seconds
        freq_hz (NDArray[np.floating]): Frequency of each channel, in Hz
        geometry (SegmentGeometry): Predicted delay-rates of the objects and the field's rate guard
        options (DelayCleanOptions | None, optional): CLEAN settings. Defaults to None (the defaults).
        visibility_noise (float | None, optional): Noise of one visibility, per real or imaginary part. If None it is estimated from the rows' spectra. Defaults to None.
        rate_margin_lobes (float, optional): Widest margin around each object's band, in units of 1/T. Defaults to 8.0.
        min_margin_lobes (float, optional): Margin an object must clear the field by, in units of 1/T. Defaults to 4.0.
        spill_lobes (float, optional): Width beyond each object's band checked for the object spilling out of it, in units of 1/T. Defaults to 4.0.

    Returns:
        SegmentFilterResult: The filtered visibilities and the rows recovered
    """
    options = options or DelayCleanOptions()
    n_rows, n_chan, n_pol = data.shape
    if not geometry.object_active.any():
        return SegmentFilterResult(outcome=SegmentOutcome.NOTHING_TO_NULL)
    if n_rows < 2 or (~is_pad).sum() == 0:
        return SegmentFilterResult(outcome=SegmentOutcome.TOO_SHORT)

    step_s = float(np.median(np.diff(time_s)))
    duration_s = n_rows * step_s
    bands, separable = _object_rate_bands(
        geometry, freq_hz, duration_s, rate_margin_lobes, min_margin_lobes
    )
    if not separable:
        return SegmentFilterResult(outcome=SegmentOutcome.NOT_SEPARABLE)

    rates_hz = delay_axis(n_rows, step_s, options.oversample)
    in_band = np.zeros((n_chan, len(rates_hz)), dtype=bool)
    near_band = np.zeros_like(in_band)
    spill_width_hz = spill_lobes / duration_s
    field_rate_hz = (
        float(np.max(geometry.field_rate)) * np.abs(freq_hz) + 1 / duration_s
    )
    for low, high in bands:
        in_band |= (rates_hz[None, :] >= low[:, None]) & (
            rates_hz[None, :] <= high[:, None]
        )
        near_band |= (rates_hz[None, :] >= (low - spill_width_hz)[:, None]) & (
            rates_hz[None, :] <= (high + spill_width_hz)[:, None]
        )
    # Just outside the objects' bands, clear of the field
    spill = near_band & ~in_band & (np.abs(rates_hz)[None, :] > field_rate_hz[:, None])

    model = np.zeros_like(data, dtype=complex)
    rows_ok = np.ones(n_rows, dtype=bool)
    for pol in range(n_pol):
        if visibility_noise is None:
            estimates = [
                estimate
                for row in np.flatnonzero(~is_pad)
                if (
                    estimate := estimate_visibility_noise(
                        data[row, :, pol], (~flags[row, :, pol]).astype(float)
                    )
                )
                is not None
            ]
            noise = float(np.median(estimates)) if estimates else np.nan
        else:
            noise = visibility_noise
        weights = (~flags[:, :, pol]).astype(float).T
        patterns, group = np.unique(weights, axis=0, return_inverse=True)
        for index, pattern in enumerate(patterns):
            if not pattern.any():
                continue
            channels = np.flatnonzero(group.reshape(-1) == index)
            selected = clean_and_select(
                values=data[:, channels, pol].T,
                weights=pattern,
                keep=in_band[channels],
                options=options,
                visibility_noise=noise,
            )
            model[:, channels, pol] = selected.model.T
            if not selected.cleaned.all():
                rows_ok[:] = False
                continue
            # The object modelled just outside its band, above CLEAN's threshold
            # and above a fraction of the object subtracted at that row, as
            # CLEAN of a bright object leaves faint components beyond any band
            spilled = np.abs(
                _model_to_values(
                    np.where(spill[channels], selected.components, 0), n_rows
                )
            )
            floor = np.maximum(
                selected.threshold[:, None],
                options.field_edge_fraction * np.abs(selected.model),
            )
            rows_ok &= ~np.any(spilled > floor, axis=0)

    return SegmentFilterResult(
        outcome=SegmentOutcome.FILTERED,
        data=data - model,
        recovered=~is_pad & rows_ok,
    )


@dataclass
class DelayCleanStatistics:
    """Running totals over the spectra cleaned in a run. Safe to share
    between the threads that process chunks."""

    spectra: int = 0
    """Spectra passed to CLEAN"""
    noise_not_estimated: int = 0
    """Spectra left unchanged because no run of channels was long enough to estimate the noise"""
    rows_flagged: int = 0
    """Rows flagged because the subtracted emission reached the field"""
    edge_residuals: list[float] = field(default_factory=list)
    """Residual at the band edges, as a multiple of the noise, of each spectrum"""
    mid_residuals: list[float] = field(default_factory=list)
    """Residual mid-band, as a multiple of the noise, of each spectrum"""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, results: list[DelayCleanResult], rows_flagged: int) -> None:
        """Add the results of one chunk"""
        with self._lock:
            self.spectra += len(results)
            self.rows_flagged += rows_flagged
            for result in results:
                if result.edge_residual is None or result.mid_residual is None:
                    if result.delay_noise == 0:
                        self.noise_not_estimated += 1
                    continue
                self.edge_residuals.append(result.edge_residual)
                self.mid_residuals.append(result.mid_residual)


def compute_delay_clean(
    data_chunk: DataChunk,
    field_delay_s: float | NDArray[np.floating],
    options: DelayCleanOptions | None = None,
    visibility_noise: float | None = None,
    active_rows: NDArray[np.bool_] | None = None,
    statistics: DelayCleanStatistics | None = None,
    flag_overlap: bool = True,
) -> TaperedChunkResult:
    """Remove emission outside the field's delay range from every row and
    polarisation of a chunk with CLEAN in delay. Rows where the subtracted
    emission reaches the field are flagged, unless ``flag_overlap`` is False.

    Args:
        data_chunk (DataChunk): The visibilities, shaped (row, channel, polarisation). Masked values are given zero weight.
        field_delay_s (float | NDArray[np.floating]): Half-width of the field's delay range, in seconds, for all rows or for each row
        options (DelayCleanOptions | None, optional): CLEAN settings. Defaults to None (the defaults).
        visibility_noise (float | None, optional): Noise of one visibility, per real or imaginary part. If None it is estimated from each spectrum. Defaults to None.
        active_rows (NDArray[np.bool_] | None, optional): Rows to clean. Others are passed through unchanged. Defaults to None (all rows).
        statistics (DelayCleanStatistics | None, optional): Totals to add this chunk's results to. Defaults to None.

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

    rows_to_clean = (
        np.ones(number_of_rows, dtype=bool) if active_rows is None else active_rows
    )

    # Spectra with no weight are passed through and not counted
    rows, pols = np.nonzero(
        np.broadcast_to(rows_to_clean[:, None], (number_of_rows, number_of_pols))
    )
    weights = (~flags[rows, :, pols]).astype(float)
    has_weight = weights.any(axis=1)
    rows, pols, weights = rows[has_weight], pols[has_weight], weights[has_weight]
    values = data[rows, :, pols]

    # One batch per flag pattern, which fixes the delay spread function
    results_by_spectrum: dict[int, DelayCleanResult] = {}
    with quiet_logs(logging.WARNING):
        if len(rows):
            patterns, group_of_spectrum = np.unique(
                weights, axis=0, return_inverse=True
            )
            group_of_spectrum = group_of_spectrum.reshape(-1)
            for group, pattern in enumerate(patterns):
                members = np.flatnonzero(group_of_spectrum == group)
                group_results = delay_clean_spectra(
                    values=values[members],
                    weights=pattern,
                    channel_width_hz=channel_width_hz,
                    field_delay_s=field_delays[rows[members]],
                    options=options,
                    visibility_noise=visibility_noise,
                )
                results_by_spectrum.update(
                    zip(members.tolist(), group_results, strict=True)
                )

    rows_reaching_field = np.zeros(number_of_rows, dtype=bool)
    results = [results_by_spectrum[spectrum] for spectrum in range(len(rows))]
    for row, pol, result in zip(rows, pols, results, strict=True):
        data[row, :, pol] -= result.model
        rows_reaching_field[row] |= result.reaches_field

    if statistics is not None:
        statistics.record(results, rows_flagged=int(rows_reaching_field.sum()))
    if flag_overlap and rows_reaching_field.any():
        logger.info(
            f"Flagging {rows_reaching_field.sum()} of {number_of_rows} rows where the object reaches the field in delay"
        )
    flagged = rows_reaching_field & flag_overlap
    new_flags = flags | flagged[:, None, None]

    return TaperedChunkResult(
        chunk_size=data_chunk.chunk_size,
        nothing_to_do=False,
        update_data=True,
        data_chunk=replace(
            data_chunk, masked_data=np.ma.masked_array(data, mask=new_flags)
        ),
        update_flags=bool(flagged.any()),
        flags=new_flags,
        overlap_rows=rows_reaching_field,
    )


class CleanTractorOptions(TractorOptions, RateFilterOptions):
    """Options for removing objects by CLEAN in delay"""

    field_delay_ns: float = 10.0
    """Half-width of the field's delay range in nanoseconds, before the guard band is added. Emission inside it is never subtracted."""
    visibility_noise: float | None = None
    """Noise of one visibility, per real or imaginary part, in the units of the data. If None it is estimated from each spectrum."""
    threshold_sigma: float = 3.0
    """CLEAN stops when the residual falls below this many times the noise"""
    mask_sigma: float = 5.0
    """The CLEAN mask grows into delays brighter than this many times the noise"""


def _active_rows(
    data_chunk: DataChunk,
    w_delays_list: list[WDelays],
    clean_tractor_options: CleanTractorOptions,
) -> tuple[list[NDArray[np.bool_]], NDArray[np.floating]]:
    """Cross-correlation rows where each object is above the elevation cut and
    within the allowed Nyquist zone, and the field's delay range of each row in
    seconds"""
    freq_hz = data_chunk.freq_chan.to("Hz").value
    half_period_s = 0.5 / abs(float(np.mean(np.diff(freq_hz))))
    cross = data_chunk.ant_1 != data_chunk.ant_2

    active = []
    field_delay_s = np.full(
        len(data_chunk.ant_1), clean_tractor_options.field_delay_ns * 1e-9
    )
    for index, w_delays in enumerate(w_delays_list):
        baseline_idx, time_idx = _get_baseline_time_indicies(w_delays, data_chunk)
        object_s = w_delays.w_delays[baseline_idx, time_idx].to("s").value
        above = (
            w_delays.elevation[time_idx].to("deg").value
            >= clean_tractor_options.elevation_cut_deg
        )
        in_zone = (
            calculate_nyquist_zone(values=object_s, upper_limit=half_period_s)
            <= clean_tractor_options.ignore_nyquist_zone
        )
        active.append(above & in_zone & cross)
        # The guard band depends only on the baseline, so is the same for every object
        if index == 0 and w_delays.guard_region is not None:
            field_delay_s = (
                field_delay_s
                + w_delays.guard_region[baseline_idx, time_idx].to("s").value
            )
    return active, field_delay_s


def compute_clean_tractor(
    data_chunk: DataChunk,
    clean_tractor_options: CleanTractorOptions,
    w_delays_list: list[WDelays],
    statistics: DelayCleanStatistics | None = None,
) -> TaperedChunkResult:
    """CLEAN the rows of a chunk where an object to null is above the
    elevation cut, and subtract the emission outside the field's delay range.

    Args:
        data_chunk (DataChunk): The visibilities, shaped (row, channel, polarisation)
        clean_tractor_options (CleanTractorOptions): The CLEAN settings
        w_delays_list (list[WDelays]): The delays of each object to null, and the guard band
        statistics (DelayCleanStatistics | None, optional): Totals to add this chunk's results to. Defaults to None.

    Returns:
        TaperedChunkResult: The cleaned visibilities and any new flags
    """
    object_active, field_delay_s = _active_rows(
        data_chunk=data_chunk,
        w_delays_list=w_delays_list,
        clean_tractor_options=clean_tractor_options,
    )
    active = np.logical_or.reduce(object_active)
    if not active.any():
        result = TaperedChunkResult(
            chunk_size=data_chunk.chunk_size,
            nothing_to_do=True,
            data_chunk=data_chunk,
        )
    else:
        result = compute_delay_clean(
            data_chunk=data_chunk,
            field_delay_s=field_delay_s,
            options=_clean_options(clean_tractor_options),
            visibility_noise=clean_tractor_options.visibility_noise,
            active_rows=active,
            statistics=statistics,
            flag_overlap=clean_tractor_options.flag_overlap,
        )
    if not clean_tractor_options.rate_filter:
        return result
    # The rate filter starts from the delay-cleaned rows with their original flags
    assert result.data_chunk is not None
    overlap = (
        np.zeros(len(active), dtype=bool)
        if result.overlap_rows is None
        else result.overlap_rows
    )
    return replace(
        result,
        rate_payload=make_rate_filter_payload(
            data_chunk=data_chunk,
            original_masked_data=np.ma.masked_array(
                result.data_chunk.masked_data.data,
                mask=np.ma.getmaskarray(data_chunk.masked_data),
            ),
            w_delays_list=w_delays_list,
            object_contaminated=[overlap & rows for rows in object_active],
        ),
    )


def _clean_options(clean_tractor_options: CleanTractorOptions) -> DelayCleanOptions:
    return DelayCleanOptions(
        mask_sigma=clean_tractor_options.mask_sigma,
        threshold_sigma=clean_tractor_options.threshold_sigma,
    )


def make_clean_rate_filter(
    open_ms_tables: OpenMSTables,
    clean_tractor_options: CleanTractorOptions,
    w_delays_list: list[WDelays],
) -> RateFilterRun:
    """Delay-rate filtering by CLEAN in fringe rate, with the delay CLEAN's settings

    Args:
        open_ms_tables (OpenMSTables): The open measurement set
        clean_tractor_options (CleanTractorOptions): The CLEAN settings
        w_delays_list (list[WDelays]): The delays of each object to null

    Returns:
        RateFilterRun: The state for ``run_tractor_chunks``
    """
    plot_dir = None
    if clean_tractor_options.make_plots:
        plot_dir = open_ms_tables.ms_path.parent / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
    settings = RateFilterSettings(
        w_delays_list=w_delays_list,
        freq_hz=np.atleast_1d(
            open_ms_tables.spw_table.getcol("CHAN_FREQ").squeeze()
        ).astype(float),
        segment_filter=partial(
            rate_clean_segment,
            options=_clean_options(clean_tractor_options),
            visibility_noise=clean_tractor_options.visibility_noise,
        ),
        field_delay_s=clean_tractor_options.field_delay_ns * 1e-9,
        elevation_cut_deg=clean_tractor_options.elevation_cut_deg,
        ignore_nyquist_zone=clean_tractor_options.ignore_nyquist_zone,
        plot_dir=plot_dir,
        max_plots=clean_tractor_options.number_of_plots,
    )
    return make_rate_filter_run(settings=settings)


def describe_clean_run(
    statistics: DelayCleanStatistics, clean_tractor_options: CleanTractorOptions
) -> str:
    """A plain description of a delay CLEAN run, for logs and the HISTORY table"""
    objects = ", ".join(clean_tractor_options.target_objects)
    guard = " plus the guard band" if clean_tractor_options.guard_field else ""
    noise = (
        f"the noise was given as {clean_tractor_options.visibility_noise:.3g}"
        if clean_tractor_options.visibility_noise is not None
        else "the noise was estimated from each spectrum"
    )
    skipped = (
        f" {statistics.noise_not_estimated} spectra were left unchanged because too few consecutive channels were unflagged to estimate the noise."
        if statistics.noise_not_estimated
        else ""
    )
    return (
        f"Delay CLEAN: removed emission outside the field's delay range "
        f"(+/-{clean_tractor_options.field_delay_ns:g} ns{guard}) from "
        f"{statistics.spectra} spectra where {objects} was above the elevation "
        f"cut; {noise}. {statistics.rows_flagged} rows were "
        f"{'flagged' if clean_tractor_options.flag_overlap else 'left unflagged (--no-flag-overlap)'}"
        f" where the subtracted emission reached the field.{skipped}"
    )


def describe_band_edges(statistics: DelayCleanStatistics) -> str | None:
    """A plain description of the residual left at the band edges, or None
    if no spectrum was long enough to measure it"""
    if not statistics.edge_residuals:
        return None
    edge = np.asarray(statistics.edge_residuals)
    mid = np.asarray(statistics.mid_residuals)
    return (
        f"Delay CLEAN residual: in the {EDGE_CHANNELS} channels at each end of "
        f"the band it is a median {np.median(edge):.2g} times the noise (95% of "
        f"spectra below {np.percentile(edge, 95):.2g}), against "
        f"{np.median(mid):.2g} in the rest of the band. Values well above 1 "
        "mean emission is left where the band cuts it off."
    )


def describe_clean_line_limit(number_of_channels: int, threshold_sigma: float) -> str:
    """The brightness above which spectral lines in the field are partly
    subtracted by delay CLEAN.

    A Gaussian line of peak S and width (FWHM) of n channels spreads over
    delay with a peak of about 1.06 S n / N in a spectrum of N channels, while
    CLEAN stops at threshold_sigma times the delay noise, sigma / sqrt(N). So
    lines are partly subtracted once S / sigma exceeds about
    threshold_sigma sqrt(N) / (1.06 n).
    """
    one_channel = threshold_sigma * np.sqrt(number_of_channels) / 1.06
    return (
        "Spectral lines in the field are partly subtracted by delay CLEAN when "
        f"their peak exceeds about {one_channel:.0f}/n times the noise per "
        f"channel, for a line n channels wide (FWHM): {one_channel:.0f} times "
        f"for 1 channel, {one_channel / 10:.0f} times for 10 channels."
    )


@dataclass
class CleanTractorResults:
    """Summary of a delay CLEAN run"""

    ms_path: Path
    """Path to the measurement set that was modified"""
    output_column: str
    """The column holding the cleaned visibilities"""
    output_plots: list[Path] | None = None
    """The output plots that were created, if any"""
    statistics: DelayCleanStatistics | None = None
    """Totals over the spectra that were cleaned. None for a dry run."""


def clean_tractor(
    ms_path: Path, clean_tractor_options: CleanTractorOptions
) -> CleanTractorResults:
    """Remove objects outside the field from a measurement set with CLEAN in
    delay. Only rows where an object is above the elevation cut are cleaned.

    Args:
        ms_path (Path): The measurement set to modify
        clean_tractor_options (CleanTractorOptions): The settings to use

    Returns:
        CleanTractorResults: A summary of the run
    """
    # Fail before touching the measurement set if rm-lite is missing
    _import_minor_loop()
    log_jolly_roger_version()
    log_dataclass_attributes(
        to_log=clean_tractor_options, class_name="CleanTractorOptions"
    )

    setup = prepare_tractor(ms_path=ms_path, tractor_options=clean_tractor_options)
    open_ms_tables = setup.open_ms_tables

    number_of_channels = len(
        np.atleast_1d(open_ms_tables.spw_table.getcol("CHAN_FREQ").squeeze())
    )
    line_limit = describe_clean_line_limit(
        number_of_channels=number_of_channels,
        threshold_sigma=clean_tractor_options.threshold_sigma,
    )
    logger.info(line_limit)

    before_flags = (
        read_baseline_flags(
            open_ms_tables,
            get_plot_baselines(open_ms_tables, clean_tractor_options.number_of_plots),
        )
        if clean_tractor_options.make_plots
        else None
    )
    statistics: DelayCleanStatistics | None = None
    rate_filter: RateFilterRun | None = None
    if not clean_tractor_options.dry_run:
        statistics = DelayCleanStatistics()
        if clean_tractor_options.rate_filter:
            rate_filter = make_clean_rate_filter(
                open_ms_tables=open_ms_tables,
                clean_tractor_options=clean_tractor_options,
                w_delays_list=setup.w_delays_list,
            )
        run_tractor_chunks(
            setup=setup,
            chunk_func=partial(
                compute_clean_tractor,
                clean_tractor_options=clean_tractor_options,
                w_delays_list=setup.w_delays_list,
                statistics=statistics,
            ),
            tractor_options=clean_tractor_options,
            rate_filter=rate_filter,
        )

    plot_paths: list[Path] | None = None
    if clean_tractor_options.make_plots:
        plot_paths = make_plot_results(
            open_ms_tables=open_ms_tables,
            data_column=clean_tractor_options.data_column,
            output_column=clean_tractor_options.output_column,
            w_delays=setup.w_delays_list,
            outer_width_ns=clean_tractor_options.field_delay_ns,
            max_baselines=clean_tractor_options.number_of_plots,
            delay_rate=clean_tractor_options.rate_filter,
            before_flags=before_flags,
        )
        logger.info(f"Made {len(plot_paths)} output plots")

    if statistics is not None:
        messages = [describe_clean_run(statistics, clean_tractor_options), line_limit]
        band_edges = describe_band_edges(statistics)
        if band_edges is not None:
            messages.append(band_edges)
        if rate_filter is not None:
            messages.append(describe_rate_filter(rate_filter))
        for message in messages:
            logger.info(message)
        add_history(
            ms_path=open_ms_tables.ms_path,
            messages=messages,
            origin="jolly_tractor clean",
            app_params=dataclass_to_app_params(clean_tractor_options),
        )

    return CleanTractorResults(
        ms_path=open_ms_tables.ms_path,
        output_column=clean_tractor_options.output_column,
        output_plots=plot_paths,
        statistics=statistics,
    )
