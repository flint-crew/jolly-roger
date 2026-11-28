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
from astropy.constants import c as speed_of_light
from numpy.typing import NDArray
from scipy import ndimage, signal, stats

from jolly_roger.baselines import OpenMSTables
from jolly_roger.history import add_history, dataclass_to_app_params
from jolly_roger.lines import (
    KAPPA,
    LineImpactSummary,
    describe_line_impact,
    summarise_line_impact,
)
from jolly_roger.logging import logger
from jolly_roger.rates import (
    RateFilterRun,
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
    make_rate_filter_settings,
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
    object_margin_resolutions: float = 3.0
    """Margin, in units of 1 / bandwidth, added either side of each object's delay band, which covers CLEAN's spread of an object between delay samples and the object's spectral structure"""


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
    """Whether the object and field may overlap: an object's band meets the field's delay range or, without object delays, subtracted emission lies just outside it"""
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


def _object_band_mask(
    delays_s: NDArray[np.floating],
    object_delay_s: NDArray[np.floating],
    object_halfwidth_s: NDArray[np.floating],
    field_delay_s: NDArray[np.floating],
    period_s: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Delays within any object's band and outside the field, shape (spectrum,
    delay), and whether any object's band meets the field, per spectrum.

    Delays wrap with ``period_s``. Objects with a NaN delay are ignored.

    Args:
        delays_s (NDArray[np.floating]): The delay axis, in seconds
        object_delay_s (NDArray[np.floating]): Each object's delay, shape (spectrum, object)
        object_halfwidth_s (NDArray[np.floating]): Each object's band half-width, shape (spectrum, object)
        field_delay_s (NDArray[np.floating]): Half-width of the field's delay range, per spectrum
        period_s (float): The period of the delay axis, 1 / channel width

    Returns:
        tuple[NDArray[np.bool_], NDArray[np.bool_]]: The delays to subtract, and whether the object meets the field
    """
    half = period_s / 2
    offset = (
        delays_s[None, None, :] - object_delay_s[..., None] + half
    ) % period_s - half
    in_band = np.abs(offset) <= object_halfwidth_s[..., None]
    in_band &= ~np.isnan(object_delay_s)[..., None]
    wrapped = np.abs((object_delay_s + half) % period_s - half)
    meets_field = np.any(
        wrapped - object_halfwidth_s <= field_delay_s[:, None],
        axis=1,
        where=~np.isnan(object_delay_s),
    )
    keep = np.any(in_band, axis=1) & (
        np.abs(delays_s)[None, :] > field_delay_s[:, None]
    )
    return keep, meets_field


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
        SelectedClean: The model of the kept components, and which series were CLEANed
    """
    options = options or DelayCleanOptions()
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    number_of_series, number_of_samples = values.shape
    size = options.oversample * number_of_samples
    result = SelectedClean(
        model=np.zeros_like(values, dtype=complex),
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
    components, _, iterations = _clean_batch(
        values=values[to_clean], weights=weights, noise=noise[to_clean], options=options
    )
    keep = np.broadcast_to(keep, (number_of_series, size))[to_clean]
    result.model[to_clean] = _model_to_values(
        np.where(keep, components, 0), number_of_samples
    )
    result.cleaned[to_clean] = iterations < options.max_iter
    return result


def delay_clean_spectra(
    values: NDArray[np.complexfloating],
    weights: NDArray[np.floating],
    channel_width_hz: float,
    field_delay_s: float | NDArray[np.floating],
    options: DelayCleanOptions | None = None,
    visibility_noise: float | NDArray[np.floating] | None = None,
    object_delay_s: NDArray[np.floating] | None = None,
    object_extent_s: NDArray[np.floating] | None = None,
) -> list[DelayCleanResult]:
    """CLEAN spectra that share their channel weights in one batch, as ``delay_clean_spectrum`` does each.

    Args:
        values (NDArray[np.complexfloating]): Visibilities, shaped (spectrum, channel), of regularly spaced channels
        weights (NDArray[np.floating]): Weight of each channel, shared by every spectrum. Zero for flagged or excluded channels.
        channel_width_hz (float): Channel width, in Hz
        field_delay_s (float | NDArray[np.floating]): Half-width of the field's delay range, in seconds, for all spectra or for each. Emission within it is kept.
        options (DelayCleanOptions | None, optional): CLEAN settings. Defaults to None (the defaults).
        visibility_noise (float | NDArray[np.floating] | None, optional): Noise of one visibility, per real or imaginary part, at unit weight, for all spectra or for each. NaN means the noise is not known, so the spectrum is not cleaned. If None it is estimated from each spectrum with ``estimate_visibility_noise``. Defaults to None.
        object_delay_s (NDArray[np.floating] | None, optional): Each object's predicted delay, shape (spectrum, object), NaN where an object is not nulled. Only components within each object's band (its delay +/- extent + margin) are subtracted, and the object meets the field where its band does. If None, everything outside the field is subtracted. Defaults to None.
        object_extent_s (NDArray[np.floating] | None, optional): Half-width of each object's disc in delay, shape (spectrum, object). Defaults to None (points).

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

    meets_field = None
    if object_delay_s is None:
        subtract = np.abs(delays_s)[None, :] > field_delays[to_clean, None]
    else:
        object_delays = np.atleast_2d(np.asarray(object_delay_s, dtype=float))
        extents = (
            np.zeros_like(object_delays)
            if object_extent_s is None
            else np.atleast_2d(np.asarray(object_extent_s, dtype=float))
        )
        resolution_s = 1 / (number_of_channels * abs(channel_width_hz))
        subtract, meets_field = _object_band_mask(
            delays_s=delays_s,
            object_delay_s=object_delays[to_clean],
            object_halfwidth_s=extents[to_clean]
            + options.object_margin_resolutions * resolution_s,
            field_delay_s=field_delays[to_clean],
            period_s=1 / abs(channel_width_hz),
        )
    model_values = _model_to_values(np.where(subtract, models, 0), number_of_channels)
    residuals = values_to_clean - _model_to_values(models, number_of_channels)
    edge_s = options.field_edge_resolutions / (
        number_of_channels * abs(channel_width_hz)
    )

    for position, spectrum in enumerate(to_clean):
        reaches_field = (
            _reaches_field(
                model=models[position],
                delays_s=delays_s,
                field_delay_s=float(field_delays[spectrum]),
                edge_s=edge_s,
                oversample=options.oversample,
                floor=options.mask_sigma * delay_noise[position],
                fraction=options.field_edge_fraction,
            )
            if meets_field is None
            else bool(meets_field[position])
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
    """Each active object's fringe-rate band in each channel, including its
    disc's extent in rate, widened by up to
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
        extent = (
            0.0
            if geometry.object_rate_extent is None
            else geometry.object_rate_extent[index][active]
        )
        low = np.min(rate - extent) * frequency
        high = np.max(rate + extent) * frequency
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
) -> SegmentFilterResult:
    """Remove the objects from a baseline segment by CLEAN in fringe rate.

    Each channel and polarisation is CLEANed along time, so the field and the
    objects are separated by fringe rate alone and the objects' delay sidelobes
    never arise. Only components in each object's fringe-rate band are
    subtracted, evaluated at every row; the residual and everything else are
    kept. No row is recovered if CLEAN did not run or converge in any channel.

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
    for low, high in bands:
        in_band |= (rates_hz[None, :] >= low[:, None]) & (
            rates_hz[None, :] <= high[:, None]
        )

    model = np.zeros_like(data, dtype=complex)
    converged = True
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
            converged &= bool(selected.cleaned.all())

    return SegmentFilterResult(
        outcome=SegmentOutcome.FILTERED,
        data=data - model,
        recovered=~is_pad & converged,
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
    object_delay_s: NDArray[np.floating] | None = None,
    object_extent_s: NDArray[np.floating] | None = None,
    protected_channels: NDArray[np.bool_] | None = None,
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
        flag_overlap (bool, optional): Flag rows where the object reaches the field. Defaults to True.
        object_delay_s (NDArray[np.floating] | None, optional): Each object's delay, shape (row, object), NaN where it is not nulled, as for ``delay_clean_spectra``. Defaults to None (everything outside the field is subtracted).
        object_extent_s (NDArray[np.floating] | None, optional): Half-width of each object's disc in delay, shape (row, object). Defaults to None (points).
        protected_channels (NDArray[np.bool_] | None, optional): Channels left out of CLEAN, e.g. spectral lines. The model is still subtracted from them. Defaults to None.

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
    if protected_channels is not None:
        weights[:, protected_channels] = 0.0
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
                    object_delay_s=None
                    if object_delay_s is None
                    else object_delay_s[rows[members]],
                    object_extent_s=None
                    if object_extent_s is None
                    else object_extent_s[rows[members]],
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
    protect_mhz: tuple[float, ...] | None = None
    """Frequency ranges in MHz, as pairs of low and high, left out of CLEAN to protect known spectral lines. The objects are still subtracted from them."""


@dataclass
class ChunkObjects:
    """Where each object is nulled in the rows of a chunk"""

    active: list[NDArray[np.bool_]]
    """Cross-correlation rows where each object is above the elevation cut and within the allowed Nyquist zone"""
    field_delay_s: NDArray[np.floating]
    """Half-width of the field's delay range of each row, in seconds"""
    object_delay_s: NDArray[np.floating]
    """Each object's delay, shape (row, object), NaN where it is not active"""
    object_extent_s: NDArray[np.floating]
    """Half-width of each object in delay, its disc plus the beam's spread, shape (row, object)"""
    meets_field: list[NDArray[np.bool_]]
    """Active rows where each object's band, with the CLEAN margin, meets the field"""


def _chunk_objects(
    data_chunk: DataChunk,
    w_delays_list: list[WDelays],
    clean_tractor_options: CleanTractorOptions,
    beam_delay_s: float = 0.0,
) -> ChunkObjects:
    """Each object's delay, extent and activity over the rows of a chunk. The
    extent is the object's disc plus ``beam_delay_s``, the spread in delay of an
    object seen through the frequency-dependent primary beam (D / c)."""
    freq_hz = data_chunk.freq_chan.to("Hz").value
    channel_width_hz = abs(float(np.mean(np.diff(freq_hz))))
    period_s = 1 / channel_width_hz
    margin_s = _clean_options(clean_tractor_options).object_margin_resolutions / (
        len(freq_hz) * channel_width_hz
    )
    cross = data_chunk.ant_1 != data_chunk.ant_2
    n_rows = len(data_chunk.ant_1)

    field_delay_s = np.full(n_rows, clean_tractor_options.field_delay_ns * 1e-9)
    active, meets_field, delays, extents = [], [], [], []
    for index, w_delays in enumerate(w_delays_list):
        baseline_idx, time_idx = _get_baseline_time_indicies(w_delays, data_chunk)
        object_s = w_delays.w_delays[baseline_idx, time_idx].to("s").value
        # The guard band depends only on the baseline, so is the same for every object
        if index == 0 and w_delays.guard_region is not None:
            field_delay_s = (
                field_delay_s
                + w_delays.guard_region[baseline_idx, time_idx].to("s").value
            )
        above = (
            w_delays.elevation[time_idx].to("deg").value
            >= clean_tractor_options.elevation_cut_deg
        )
        in_zone = (
            calculate_nyquist_zone(values=object_s, upper_limit=period_s / 2)
            <= clean_tractor_options.ignore_nyquist_zone
        )
        is_active = above & in_zone & cross
        extent_s = beam_delay_s + (
            np.zeros(n_rows)
            if w_delays.object_extent is None
            else w_delays.object_extent[baseline_idx, time_idx].to("s").value
        )
        wrapped_s = np.abs((object_s + period_s / 2) % period_s - period_s / 2)
        active.append(is_active)
        meets_field.append(
            is_active & (wrapped_s - extent_s - margin_s <= field_delay_s)
        )
        delays.append(np.where(is_active, object_s, np.nan))
        extents.append(extent_s)
    return ChunkObjects(
        active=active,
        field_delay_s=field_delay_s,
        object_delay_s=np.stack(delays, axis=1),
        object_extent_s=np.stack(extents, axis=1),
        meets_field=meets_field,
    )


def protected_channel_mask(
    freq_hz: NDArray[np.floating], protect_mhz: tuple[float, ...] | None
) -> NDArray[np.bool_] | None:
    """Channels within any of the (low, high) frequency ranges, in MHz, given
    as a flat sequence of pairs. None if there are none."""
    if not protect_mhz:
        return None
    if len(protect_mhz) % 2:
        msg = f"protect_mhz must be pairs of (low, high) in MHz, got {protect_mhz}"
        raise ValueError(msg)
    freq_mhz = np.asarray(freq_hz, dtype=float) * 1e-6
    protected = np.zeros(len(freq_mhz), dtype=bool)
    for low, high in zip(protect_mhz[::2], protect_mhz[1::2], strict=True):
        protected |= (freq_mhz >= min(low, high)) & (freq_mhz <= max(low, high))
    return protected


def compute_clean_tractor(
    data_chunk: DataChunk,
    clean_tractor_options: CleanTractorOptions,
    w_delays_list: list[WDelays],
    statistics: DelayCleanStatistics | None = None,
    beam_delay_s: float = 0.0,
) -> TaperedChunkResult:
    """CLEAN the rows of a chunk where an object to null is above the
    elevation cut, and subtract the emission within each object's delay band.

    Args:
        data_chunk (DataChunk): The visibilities, shaped (row, channel, polarisation)
        clean_tractor_options (CleanTractorOptions): The CLEAN settings
        w_delays_list (list[WDelays]): The delays and extent of each object to null, and the guard band
        statistics (DelayCleanStatistics | None, optional): Totals to add this chunk's results to. Defaults to None.
        beam_delay_s (float, optional): Spread in delay of an object seen through the primary beam, D / c, added to each object's band. Defaults to 0.0.

    Returns:
        TaperedChunkResult: The cleaned visibilities and any new flags
    """
    objects = _chunk_objects(
        data_chunk=data_chunk,
        w_delays_list=w_delays_list,
        clean_tractor_options=clean_tractor_options,
        beam_delay_s=beam_delay_s,
    )
    active = np.logical_or.reduce(objects.active)
    if not active.any():
        result = TaperedChunkResult(
            chunk_size=data_chunk.chunk_size,
            nothing_to_do=True,
            data_chunk=data_chunk,
        )
    else:
        result = compute_delay_clean(
            data_chunk=data_chunk,
            field_delay_s=objects.field_delay_s,
            options=_clean_options(clean_tractor_options),
            visibility_noise=clean_tractor_options.visibility_noise,
            active_rows=active,
            statistics=statistics,
            flag_overlap=clean_tractor_options.flag_overlap,
            object_delay_s=objects.object_delay_s,
            object_extent_s=objects.object_extent_s,
            protected_channels=protected_channel_mask(
                data_chunk.freq_chan.to("Hz").value, clean_tractor_options.protect_mhz
            ),
        )
    if not clean_tractor_options.rate_filter:
        return result
    # The rate filter starts from the delay-cleaned rows with their original flags
    assert result.data_chunk is not None
    return replace(
        result,
        rate_payload=make_rate_filter_payload(
            data_chunk=data_chunk,
            original_masked_data=np.ma.masked_array(
                result.data_chunk.masked_data.data,
                mask=np.ma.getmaskarray(data_chunk.masked_data),
            ),
            w_delays_list=w_delays_list,
            object_contaminated=objects.meets_field,
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
    return make_rate_filter_run(
        settings=make_rate_filter_settings(
            open_ms_tables=open_ms_tables,
            tractor_options=clean_tractor_options,
            w_delays_list=w_delays_list,
            segment_filter=partial(
                rate_clean_segment,
                options=_clean_options(clean_tractor_options),
                visibility_noise=clean_tractor_options.visibility_noise,
            ),
            field_delay_s=clean_tractor_options.field_delay_ns * 1e-9,
        )
    )


def describe_clean_run(
    statistics: DelayCleanStatistics,
    clean_tractor_options: CleanTractorOptions,
    w_delays_list: list[WDelays] | None = None,
    beam_delay_s: float = 0.0,
) -> str:
    """A plain description of a delay CLEAN run, for logs and the HISTORY table"""
    objects = ", ".join(clean_tractor_options.target_objects)
    if w_delays_list is not None:
        objects = ", ".join(
            f"{w_delays.object_name} (radius {w_delays.object_radius.to_value('deg'):.3g} deg)"
            for w_delays in w_delays_list
        )
    margin = _clean_options(clean_tractor_options).object_margin_resolutions
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
        f"Delay CLEAN: removed emission within each object's delay band (its "
        f"delay +/- its disc's extent + {beam_delay_s * 1e9:.3g} ns for the primary "
        f"beam (D/c) + {margin:g}/bandwidth), outside the field's "
        f"delay range (+/-{clean_tractor_options.field_delay_ns:g} ns{guard}), from "
        f"{statistics.spectra} spectra where {objects} was above the elevation "
        f"cut; {noise}. {statistics.rows_flagged} rows were "
        f"{'flagged' if clean_tractor_options.flag_overlap else 'left unflagged (--no-flag-overlap)'}"
        f" where an object's band met the field.{skipped}"
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


def describe_clean_line_limit(number_of_channels: int, mask_sigma: float) -> str:
    """The brightness below which delay CLEAN leaves spectral lines in the field
    untouched.

    A Gaussian line of peak S and FWHM of n channels spreads over delay with a
    peak of kappa S n / N in a spectrum of N channels (kappa = 1.06). CLEAN
    only reaches it once that exceeds the mask, mask_sigma times the delay
    noise sigma / sqrt(N): S / sigma > mask_sigma sqrt(N) / (kappa n). Above
    that, a line loses the part of its delay spread inside the objects' bands,
    as ``report_clean_line_impact`` gives.
    """
    one_channel = mask_sigma * np.sqrt(number_of_channels) / KAPPA
    return (
        "Delay CLEAN leaves spectral lines in the field untouched while their peak "
        f"is below about {one_channel:.0f}/n times the noise per channel, for a "
        f"line n channels wide (FWHM): {one_channel:.0f} times for 1 channel, "
        f"{one_channel / 10:.0f} times for 10 channels. Brighter lines lose the "
        "part of their delay spread inside each object's band."
    )


def report_clean_line_impact(
    open_ms_tables: OpenMSTables,
    clean_tractor_options: CleanTractorOptions,
    w_delays_list: list[WDelays],
    beam_delay_s: float = 0.0,
) -> dict[str, LineImpactSummary]:
    """The largest loss of a bright spectral line in the field, from geometry,
    for each object, over the baselines in the measurement set. CLEAN subtracts
    a band of width 2 (extent + margin) about the object's delay, so a line
    loses as it would to a notch of that width (``summarise_line_impact``).
    Rows where the band meets the field are flagged and not counted.

    Args:
        open_ms_tables (OpenMSTables): The open measurement set
        clean_tractor_options (CleanTractorOptions): The CLEAN settings
        w_delays_list (list[WDelays]): The delays and extent of each object
        beam_delay_s (float, optional): Spread in delay from the primary beam, D / c, added to each object's band. Defaults to 0.0.

    Returns:
        dict[str, LineImpactSummary]: The summary for each object, keyed by name
    """
    main_table = open_ms_tables.main_table
    present = {
        (int(ant_1), int(ant_2))
        for ant_1, ant_2 in zip(
            main_table.getcol("ANTENNA1"), main_table.getcol("ANTENNA2"), strict=True
        )
        if ant_1 != ant_2
    }
    freq_hz = np.atleast_1d(open_ms_tables.spw_table.getcol("CHAN_FREQ").squeeze())
    channel_width_hz = abs(float(np.mean(np.diff(freq_hz))))
    period_s = 1 / channel_width_hz
    margin_s = _clean_options(clean_tractor_options).object_margin_resolutions / (
        len(freq_hz) * channel_width_hz
    )
    field_s = clean_tractor_options.field_delay_ns * 1e-9
    summaries = {}
    for w_delays in w_delays_list:
        baselines = [index for pair, index in w_delays.b_map.items() if pair in present]
        object_s = w_delays.w_delays[baselines].to("s").value
        extent_s = beam_delay_s + (
            np.zeros_like(object_s)
            if w_delays.object_extent is None
            else w_delays.object_extent[baselines].to("s").value
        )
        guard_s = (
            np.zeros_like(object_s)
            if w_delays.guard_region is None
            else w_delays.guard_region[baselines].to("s").value
        )
        above = (
            w_delays.elevation.to("deg").value
            >= clean_tractor_options.elevation_cut_deg
        )[None, :]
        in_zone = (
            calculate_nyquist_zone(values=object_s, upper_limit=period_s / 2)
            <= clean_tractor_options.ignore_nyquist_zone
        )
        separation_s = np.abs((object_s + period_s / 2) % period_s - period_s / 2)
        half_width_s = extent_s + margin_s
        cleaned = above & in_zone & (separation_s - half_width_s > field_s + guard_s)
        summaries[w_delays.object_name] = summarise_line_impact(
            delay_separation_s=separation_s[cleaned],
            effective_width_s=2 * half_width_s[cleaned],
            channel_width_hz=channel_width_hz,
            reference_frequency_hz=float(np.mean(freq_hz)),
        )
    return summaries


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
    beam_delay_s = 0.0
    if setup.dish_diameter_m is None:
        logger.warning(
            "The dish diameter is unknown, so the objects' delay bands leave out the "
            "spread from the primary beam (D/c) and may be too narrow. Set "
            "--dish-diameter-m."
        )
    else:
        beam_delay_s = setup.dish_diameter_m / speed_of_light.to_value("m/s")

    number_of_channels = len(
        np.atleast_1d(open_ms_tables.spw_table.getcol("CHAN_FREQ").squeeze())
    )
    line_messages = [
        describe_clean_line_limit(
            number_of_channels=number_of_channels,
            mask_sigma=clean_tractor_options.mask_sigma,
        ),
        *(
            describe_line_impact(name, summary)
            for name, summary in report_clean_line_impact(
                open_ms_tables=open_ms_tables,
                clean_tractor_options=clean_tractor_options,
                w_delays_list=setup.w_delays_list,
                beam_delay_s=beam_delay_s,
            ).items()
        ),
    ]
    for message in line_messages:
        logger.info(message)

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
                beam_delay_s=beam_delay_s,
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
        messages = [
            describe_clean_run(
                statistics, clean_tractor_options, setup.w_delays_list, beam_delay_s
            ),
            *line_messages,
        ]
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
