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
import threading
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jolly_roger.history import add_history, dataclass_to_app_params
from jolly_roger.logging import logger
from jolly_roger.tractor import (
    DataChunk,
    TaperedChunkResult,
    TractorOptions,
    _get_baseline_time_indicies,
    make_plot_results,
    prepare_tractor,
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
    edge_residual, mid_residual = _residual_by_band_position(
        residual=values - _model_to_values(model, number_of_channels),
        used=weights > 0,
        noise=visibility_noise,
    )

    return DelayCleanResult(
        model=model_values,
        iterations=result.iter_count,
        delay_noise=delay_noise,
        reaches_field=reaches_field,
        edge_residual=edge_residual,
        mid_residual=mid_residual,
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
) -> TaperedChunkResult:
    """Remove emission outside the field's delay range from every row and
    polarisation of a chunk with CLEAN in delay. Rows where the subtracted
    emission reaches the field are flagged.

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

    rows_reaching_field = np.zeros(number_of_rows, dtype=bool)
    results: list[DelayCleanResult] = []
    with quiet_logs(logging.WARNING):
        for row in np.flatnonzero(rows_to_clean):
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
                results.append(result)

    if statistics is not None:
        statistics.record(results, rows_flagged=int(rows_reaching_field.sum()))
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


class CleanTractorOptions(TractorOptions):
    """Options for removing objects by CLEAN in delay"""

    guard_field: bool = True
    """Protect the field's delay range, sized from the field of view on each baseline, in addition to field_delay_ns. On by default, as CLEAN subtracts everything outside it; on the command line the flag turns it off."""
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
) -> tuple[NDArray[np.bool_], NDArray[np.floating]]:
    """Rows where any object is above the elevation cut and within the allowed
    Nyquist zone, and the field's delay range of each row in seconds"""
    freq_hz = data_chunk.freq_chan.to("Hz").value
    half_period_s = 0.5 / abs(float(np.mean(np.diff(freq_hz))))
    cross = data_chunk.ant_1 != data_chunk.ant_2

    active = np.zeros(len(data_chunk.ant_1), dtype=bool)
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
        active |= above & in_zone
        # The guard band depends only on the baseline, so is the same for every object
        if index == 0 and w_delays.guard_region is not None:
            field_delay_s = (
                field_delay_s
                + w_delays.guard_region[baseline_idx, time_idx].to("s").value
            )
    return active & cross, field_delay_s


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
    active, field_delay_s = _active_rows(
        data_chunk=data_chunk,
        w_delays_list=w_delays_list,
        clean_tractor_options=clean_tractor_options,
    )
    if not active.any():
        return TaperedChunkResult(
            chunk_size=data_chunk.chunk_size,
            nothing_to_do=True,
            data_chunk=data_chunk,
        )
    return compute_delay_clean(
        data_chunk=data_chunk,
        field_delay_s=field_delay_s,
        options=DelayCleanOptions(
            mask_sigma=clean_tractor_options.mask_sigma,
            threshold_sigma=clean_tractor_options.threshold_sigma,
        ),
        visibility_noise=clean_tractor_options.visibility_noise,
        active_rows=active,
        statistics=statistics,
    )


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
        f"cut; {noise}. {statistics.rows_flagged} rows were flagged because the "
        f"subtracted emission reached the field.{skipped}"
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

    statistics: DelayCleanStatistics | None = None
    if not clean_tractor_options.dry_run:
        statistics = DelayCleanStatistics()
        run_tractor_chunks(
            setup=setup,
            chunk_func=partial(
                compute_clean_tractor,
                clean_tractor_options=clean_tractor_options,
                w_delays_list=setup.w_delays_list,
                statistics=statistics,
            ),
            tractor_options=clean_tractor_options,
        )

    plot_paths: list[Path] | None = None
    if clean_tractor_options.make_plots:
        plot_paths = make_plot_results(
            open_ms_tables=open_ms_tables,
            data_column=clean_tractor_options.data_column,
            output_column=clean_tractor_options.output_column,
            w_delays=setup.w_delays_list,
            reverse_baselines=clean_tractor_options.reverse_baselines,
            outer_width_ns=clean_tractor_options.field_delay_ns,
            max_baselines=clean_tractor_options.number_of_plots,
        )
        logger.info(f"Made {len(plot_paths)} output plots")

    if statistics is not None:
        messages = [describe_clean_run(statistics, clean_tractor_options), line_limit]
        band_edges = describe_band_edges(statistics)
        if band_edges is not None:
            messages.append(band_edges)
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
