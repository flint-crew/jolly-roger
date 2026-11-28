"""Delay-rate filtering of rows the delay notch flags"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len

from jolly_roger.uvws import WDelays, _time_map_to_seconds, get_w_rates
from jolly_roger.weights import scale_multiple_weights
from jolly_roger.wrap import calculate_nyquist_zone


@dataclass
class _Row:
    """A single row of a measurement set held by the accumulator"""

    row: int
    time_mjd: float
    time_idx: int
    data: NDArray[np.complexfloating[Any]]
    mask: NDArray[np.bool_]
    weights: dict[str, NDArray[np.floating[Any]]] | None


@dataclass
class CandidateSegment:
    """Consecutive rows of a baseline to filter in delay-rate: candidate rows,
    plus clean pad rows either side that are never written back"""

    ant_1: int
    """The first antenna of the baseline"""
    ant_2: int
    """The second antenna of the baseline"""
    baseline_idx: int
    """The index of the baseline into the ``WDelays``"""
    rows: list[int] = field(default_factory=list)
    """The row number into the measurement set"""
    time_mjds: list[float] = field(default_factory=list)
    """The time of each row, MJD in seconds"""
    time_idx: list[int] = field(default_factory=list)
    """The index of each row's time into the ``WDelays``"""
    data: list[NDArray[np.complexfloating[Any]]] = field(default_factory=list)
    """The original visibilities of each row, shape (chan, pol)"""
    mask: list[NDArray[np.bool_]] = field(default_factory=list)
    """The original flags of each row, shape (chan, pol)"""
    is_pad: list[bool] = field(default_factory=list)
    """Whether each row is a pad row rather than a candidate"""
    n_candidates: int = 0
    """The number of candidate rows"""
    weights: dict[str, list[NDArray[np.floating[Any]]]] | None = None
    """The original weights of each row, keyed by column name"""
    needed_s: float = 0.0
    """The shortest duration, in seconds, that separates every object from the field over the candidates"""


def _append_segment_row(segment: CandidateSegment, row: _Row, is_pad: bool) -> None:
    """Add a row to the end of a segment"""
    segment.rows.append(row.row)
    segment.time_mjds.append(row.time_mjd)
    segment.time_idx.append(row.time_idx)
    segment.data.append(row.data)
    segment.mask.append(row.mask)
    segment.is_pad.append(is_pad)
    if not is_pad:
        segment.n_candidates += 1
    if row.weights is not None:
        if segment.weights is None:
            segment.weights = {k: [] for k in row.weights}
        for k, v in row.weights.items():
            segment.weights[k].append(v)


@dataclass
class _BaselineState:
    """Tracking of a single baseline in the accumulator"""

    segment: CandidateSegment | None = None
    """The segment being collected. None when idle."""
    trailing: int | None = None
    """Number of trailing pad rows collected. None when the segment is still open."""
    leading: deque[_Row] = field(default_factory=deque)
    """The most recent consecutive clean rows, used to lead a new segment"""
    last_time_idx: int | None = None
    """The time index of the last row seen"""
    last_time_mjd: float | None = None
    """The time of the last row seen, MJD in seconds"""


@dataclass
class SegmentAccumulator:
    """Per-baseline collection of candidate rows: rows flagged by the delay
    notch where every object is separable from the field in delay-rate. A
    segment is released on a clean row (after ``pad_timesteps`` trailing
    rows), a flagged row that is not a candidate, a gap in time, or once it
    lasts ``length_factor`` times the duration its candidates need. A segment
    released while still in a run of candidates overlaps the next by
    2 ``pad_timesteps`` rows, so every candidate is away from an end of the
    segment it is written from."""

    length_factor: float = 2.0
    """A segment is released once it lasts this many times the duration its candidates need"""
    max_timesteps: int | None = None
    """The most candidates in a segment. If None there is no limit."""
    pad_timesteps: int = 0
    """The number of clean rows either side of a segment to include as padding"""
    max_time_gap_s: float | None = None
    """The largest step in time within a segment, in seconds. If None only a skipped time index is a gap."""
    states: dict[tuple[int, int], _BaselineState] = field(default_factory=dict)
    """The tracking of each baseline, keyed by (ANTENNA1, ANTENNA2)"""
    doubly_contaminated_rows: int = 0
    """Rows flagged by the delay notch that are not separable in delay-rate either"""


@dataclass
class RateFilterPayload:
    """The per-row information needed to delay-rate filter the rows of a chunk"""

    rows: NDArray[np.int_]
    """The row number of each row in the measurement set"""
    ant_1: NDArray[np.int_]
    """The first antenna of each row"""
    ant_2: NDArray[np.int_]
    """The second antenna of each row"""
    time_mjds: NDArray[np.floating[Any]]
    """The time of each row, MJD in seconds"""
    baseline_idx: NDArray[np.int_]
    """The index of each row's baseline into the ``WDelays``"""
    time_idx: NDArray[np.int_]
    """The index of each row's time into the ``WDelays``"""
    original_masked_data: np.ma.MaskedArray
    """The visibilities before any delay tapering, shape (row, chan, pol)"""
    delay_contaminated: NDArray[np.bool_]
    """Rows flagged by the delay notch"""
    candidate: NDArray[np.bool_]
    """Flagged rows where every object is separable from the field in delay-rate"""
    needed_s: NDArray[np.floating[Any]] | None = None
    """The duration, in seconds, each row needs to separate its objects from the field. If None segments are not released on length."""
    original_weights: dict[str, NDArray[np.floating[Any]]] | None = None
    """The weights before any scaling"""


def _segment_row(segment: CandidateSegment, index: int) -> _Row:
    return _Row(
        row=segment.rows[index],
        time_mjd=segment.time_mjds[index],
        time_idx=segment.time_idx[index],
        data=segment.data[index],
        mask=segment.mask[index],
        weights=None
        if segment.weights is None
        else {k: v[index] for k, v in segment.weights.items()},
    )


def _split_segment(
    state: _BaselineState, released: list[CandidateSegment], pad: int
) -> None:
    """Release a segment in the middle of a run of candidates. Its last ``pad``
    candidates become pad rows, and the next segment starts with its last
    ``2 pad`` rows: ``pad`` pad rows, then those candidates."""
    segment = state.segment
    assert segment is not None
    if pad == 0 or segment.n_candidates <= 2 * pad:
        _release_segment(state, released)
        return
    n_rows = len(segment.rows)
    overlap = [
        _segment_row(segment, index) for index in range(n_rows - 2 * pad, n_rows)
    ]
    for index in range(n_rows - pad, n_rows):
        segment.is_pad[index] = True
    segment.n_candidates -= pad
    _release_segment(state, released)
    state.segment = CandidateSegment(
        ant_1=segment.ant_1,
        ant_2=segment.ant_2,
        baseline_idx=segment.baseline_idx,
        needed_s=segment.needed_s,
    )
    for index, row in enumerate(overlap):
        _append_segment_row(state.segment, row, is_pad=index < pad)


def _release_segment(state: _BaselineState, released: list[CandidateSegment]) -> None:
    """Move the segment of a baseline, if any, to the released segments"""
    if state.segment is not None:
        released.append(state.segment)
    state.segment = None
    state.trailing = None


def _is_time_gap(
    state: _BaselineState,
    time_idx: int,
    time_mjd: float,
    max_time_gap_s: float | None,
) -> bool:
    """Whether a row does not directly follow the previous row of its baseline"""
    if state.last_time_idx is None or state.last_time_mjd is None:
        return False
    if time_idx - state.last_time_idx != 1:
        return True
    return (
        max_time_gap_s is not None and time_mjd - state.last_time_mjd > max_time_gap_s
    )


def update_segment_accumulator(
    accumulator: SegmentAccumulator, payload: RateFilterPayload
) -> list[CandidateSegment]:
    """Consume the rows of a chunk, in the order they appear in the measurement set.

    Args:
        accumulator (SegmentAccumulator): The per-baseline collection of rows, updated in place
        payload (RateFilterPayload): The rows, and whether each is flagged and a candidate

    Returns:
        list[CandidateSegment]: Segments that have been released and are ready to filter
    """
    pad = accumulator.pad_timesteps
    max_timesteps = accumulator.max_timesteps
    data = np.ma.getdata(payload.original_masked_data)
    mask = np.ma.getmaskarray(payload.original_masked_data)
    weights = payload.original_weights

    released: list[CandidateSegment] = []
    for i in range(len(payload.rows)):
        is_candidate = bool(payload.candidate[i])
        is_clean = not bool(payload.delay_contaminated[i])
        key = (int(payload.ant_1[i]), int(payload.ant_2[i]))
        state = accumulator.states.get(key)

        if state is None:
            # Nothing is needed of idle baselines that see clean rows
            if is_clean and pad == 0:
                continue
            state = accumulator.states.setdefault(
                key, _BaselineState(leading=deque(maxlen=pad))
            )

        t_idx = int(payload.time_idx[i])
        t_mjd = float(payload.time_mjds[i])
        if _is_time_gap(state, t_idx, t_mjd, accumulator.max_time_gap_s):
            _release_segment(state, released)
            state.leading.clear()
        state.last_time_idx = t_idx
        state.last_time_mjd = t_mjd

        if not is_clean and not is_candidate:
            accumulator.doubly_contaminated_rows += 1
            _release_segment(state, released)
            state.leading.clear()
            continue

        if is_clean and pad == 0:
            # Without padding a clean row only ends a segment, so is not kept
            _release_segment(state, released)
            continue

        # Copy so the chunk the row was drawn from can be released
        row = _Row(
            row=int(payload.rows[i]),
            time_mjd=t_mjd,
            time_idx=t_idx,
            data=np.array(data[i]),
            mask=np.array(mask[i]),
            weights=None
            if weights is None
            else {k: np.array(v[i]) for k, v in weights.items()},
        )

        if is_candidate:
            if state.trailing is not None:
                _release_segment(state, released)
            if state.segment is None:
                state.segment = CandidateSegment(
                    ant_1=key[0],
                    ant_2=key[1],
                    baseline_idx=int(payload.baseline_idx[i]),
                )
                for lead in state.leading:
                    _append_segment_row(state.segment, lead, is_pad=True)
            state.leading.clear()
            _append_segment_row(state.segment, row, is_pad=False)
            segment = state.segment
            if payload.needed_s is not None:
                segment.needed_s = max(segment.needed_s, float(payload.needed_s[i]))
            duration_s = segment.time_mjds[-1] - segment.time_mjds[0]
            long_enough = (
                payload.needed_s is not None
                and duration_s >= accumulator.length_factor * segment.needed_s
            )
            if long_enough or (
                max_timesteps is not None and segment.n_candidates >= max_timesteps
            ):
                _split_segment(state, released, pad)
            continue

        # Clean row, with padding
        if state.segment is not None:
            _append_segment_row(state.segment, row, is_pad=True)
            state.trailing = (state.trailing or 0) + 1
            if state.trailing >= pad:
                _release_segment(state, released)
        state.leading.append(row)

    return released


def flush_segment_accumulator(
    accumulator: SegmentAccumulator,
) -> list[CandidateSegment]:
    """Release all segments still being collected, e.g. at the end of the measurement set

    Args:
        accumulator (SegmentAccumulator): The per-baseline collection of rows, emptied in place

    Returns:
        list[CandidateSegment]: The remaining segments
    """
    released: list[CandidateSegment] = []
    for state in accumulator.states.values():
        _release_segment(state, released)
    accumulator.states = {}
    return released


@dataclass(frozen=True)
class SegmentGeometry:
    """Predicted delays (s) and delay-rates (dimensionless) over a segment"""

    object_names: tuple[str, ...]
    """The name of each object"""
    object_delay_s: NDArray[np.floating[Any]]
    """Delay of each object, shape (object, row)"""
    object_rate: NDArray[np.floating[Any]]
    """Delay-rate of each object, shape (object, row)"""
    object_active: NDArray[np.bool_]
    """Whether each object is nulled, shape (object, row)"""
    field_delay_s: NDArray[np.floating[Any]]
    """Half-width of the field's protected delays, shape (row,)"""
    field_rate: NDArray[np.floating[Any]]
    """Half-width of the field's protected delay-rates, shape (row,)"""


class SegmentOutcome(Enum):
    """The outcome of filtering a segment"""

    FILTERED = "filtered"
    TOO_SHORT = "too short"
    NOT_SEPARABLE = "not separable"
    NOTHING_TO_NULL = "nothing to null"


@dataclass
class SegmentFilterResult:
    """The result of filtering one segment"""

    outcome: SegmentOutcome
    """What happened to the segment"""
    data: NDArray[np.complexfloating[Any]] | None = None
    """Filtered visibilities, shape (row, chan, pol). None unless filtered."""
    recovered: NDArray[np.bool_] | None = None
    """Candidates that passed the filter's checks"""
    removed_fraction: NDArray[np.floating[Any]] | None = None
    """Fraction of the rate axis removed at each delay, shape (row, delay)"""
    removal: NDArray[np.floating[Any]] | None = None
    """The region removed, shape (rate, delay) in FFT order"""
    window: NDArray[np.floating[Any]] | None = None
    """The time window applied to the rows"""


def _roll_off(
    distance: NDArray[np.floating[Any]], width: float
) -> NDArray[np.floating[Any]]:
    """One where ``distance`` <= 0, falling as a half cosine to zero over ``width``"""
    if width <= 0:
        return (distance <= 0).astype(float)
    return 0.5 * (1.0 + np.cos(np.pi * np.clip(distance / width, 0.0, 1.0)))


def _wrap(
    values: NDArray[np.floating[Any]], half_period: float
) -> NDArray[np.floating[Any]]:
    return (np.asarray(values) + half_period) % (2 * half_period) - half_period


def _outside(
    axis: NDArray[np.floating[Any]], low: float, high: float, half_period: float
) -> NDArray[np.floating[Any]]:
    """Distance beyond the interval [low, high] on a cyclic axis"""
    centre, half_width = 0.5 * (low + high), 0.5 * (high - low)
    return np.abs(_wrap(axis - centre, half_period)) - half_width


def _time_window(
    is_pad: NDArray[np.bool_], roll_off_rows: int
) -> NDArray[np.floating[Any]]:
    """One over the candidates, rolling off over the pad rows. Candidates within
    ``roll_off_rows`` of an end of the segment fall in the roll-off."""
    index = np.arange(len(is_pad))
    candidates = index[~is_pad]
    flat_start = max(candidates[0], roll_off_rows)
    flat_end = min(candidates[-1], len(is_pad) - 1 - roll_off_rows)
    distance = np.maximum(flat_start - index, index - flat_end)
    return _roll_off(distance, roll_off_rows + 1)


def _band_residual(
    data: NDArray[np.complexfloating[Any]],
    weights: NDArray[np.floating[Any]],
    rate_profile: NDArray[np.floating[Any]],
    object_window: NDArray[np.bool_],
    all_windows: NDArray[np.bool_],
    n_time: int,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Peak of an object in each row after band-passing to its fringe rates,
    and the noise from delays outside every object's window. Both (row, pol)."""
    n_rows = data.shape[0]
    spectrum = np.fft.fft(np.where(weights > 0, data, 0) * weights, n=n_time, axis=0)
    band = np.fft.ifft(spectrum * rate_profile[:, None, None], axis=0)[:n_rows]
    norm = np.maximum(np.sum(weights, axis=1), 1e-30)
    amplitude = np.abs(np.fft.fft(band, axis=1) / norm[:, None, :])
    peak = np.max(np.where(object_window[..., None], amplitude, 0.0), axis=1)
    # Median of a Rayleigh distribution is sigma * sqrt(2 ln 2)
    outside = np.where(~all_windows[..., None], amplitude, np.nan)
    noise = np.nanmedian(outside, axis=1) / np.sqrt(2 * np.log(2))
    return peak, np.nan_to_num(noise, nan=np.inf)


def filter_segment(
    data: NDArray[np.complexfloating[Any]],
    flags: NDArray[np.bool_],
    is_pad: NDArray[np.bool_],
    time_s: NDArray[np.floating[Any]],
    freq_hz: NDArray[np.floating[Any]],
    geometry: SegmentGeometry,
    delay_width_s: float,
    delay_roll_off_s: float,
    roll_off_rows: int = 16,
    rate_margin_lobes: float = 4.0,
    rate_roll_off_bins: int = 2,
    acceptance_factor: float = 2.0,
    acceptance_sigma: float = 5.0,
) -> SegmentFilterResult:
    """Remove the objects from a baseline segment in delay and fringe rate,
    keeping the field.

    The segment is windowed in time (flat over the candidates) and zero-padded
    before the 2D transform. Each object's delay track +/- ``delay_width_s``
    and fringe-rate band +/- ``rate_margin_lobes`` / T are removed, except
    inside the field's protected region. A candidate is recovered if the object
    left in it is no more than ``acceptance_factor`` times what the delay notch
    alone leaves, plus ``acceptance_sigma`` times the noise.

    Args:
        data (NDArray[np.complexfloating[Any]]): Visibilities before nulling, shape (row, chan, pol)
        flags (NDArray[np.bool_]): Flags before nulling, shape (row, chan, pol)
        is_pad (NDArray[np.bool_]): Whether each row is a pad row
        time_s (NDArray[np.floating[Any]]): Time of each row, in seconds
        freq_hz (NDArray[np.floating[Any]]): Frequency of each channel, in Hz
        geometry (SegmentGeometry): Predicted delays and rates of the objects and the field
        delay_width_s (float): Half-width removed around each object's delay track, in seconds
        delay_roll_off_s (float): Roll-off in delay, in seconds
        roll_off_rows (int, optional): Rows over which the time window rolls off. Defaults to 16.
        rate_margin_lobes (float, optional): Margin around each object's fringe-rate band, in units of 1/T. Defaults to 4.0.
        rate_roll_off_bins (int, optional): Roll-off in fringe rate, in bins. Defaults to 2.
        acceptance_factor (float, optional): Allowed residual relative to the delay notch alone. Defaults to 2.0.
        acceptance_sigma (float, optional): Allowed residual above that, in units of the noise. Defaults to 5.0.

    Returns:
        SegmentFilterResult: The filtered data and the recovered rows
    """
    n_rows, n_chan, _ = data.shape
    if not geometry.object_active.any():
        return SegmentFilterResult(outcome=SegmentOutcome.NOTHING_TO_NULL)

    window = _time_window(is_pad, roll_off_rows)
    usable = ~is_pad & (window >= 1.0)
    if usable.sum() < 2:
        return SegmentFilterResult(outcome=SegmentOutcome.TOO_SHORT)

    step_s = float(np.median(np.diff(time_s)))
    duration_s = n_rows * step_s
    n_time = int(next_fast_len(2 * n_rows))
    rates_hz = np.fft.fftfreq(n_time, step_s)
    rate_bin_hz = 1.0 / (n_time * step_s)
    rate_half_period = 0.5 / step_s
    rate_roll_off_hz = rate_roll_off_bins * rate_bin_hz
    channel_hz = float(np.mean(np.diff(freq_hz)))
    delays_s = np.fft.fftfreq(n_chan, channel_hz)
    delay_half_period = 0.5 / abs(channel_hz)
    nu_low, nu_high = np.min(np.abs(freq_hz)), np.max(np.abs(freq_hz))

    field_delay_s = float(np.max(geometry.field_delay_s))
    field_rate_hz = float(np.max(geometry.field_rate)) * nu_high + 1.0 / duration_s
    margin_hz = rate_margin_lobes / duration_s

    removal = np.zeros((n_time, n_chan))
    row_notch = np.ones((n_rows, n_chan))
    all_windows = np.zeros((n_rows, n_chan), dtype=bool)
    checks = []
    for index in range(len(geometry.object_names)):
        active = geometry.object_active[index]
        if not active.any():
            continue
        delay = geometry.object_delay_s[index]
        track = delay[active][0] + np.unwrap(
            _wrap(delay[active] - delay[active][0], delay_half_period),
            period=2 * delay_half_period,
        )
        low_s, high_s = track.min() - delay_width_s, track.max() + delay_width_s
        delay_profile = _roll_off(
            _outside(delays_s, low_s, high_s, delay_half_period), delay_roll_off_s
        )
        fringe = np.outer(geometry.object_rate[index][active], [nu_low, nu_high])
        low_hz, high_hz = fringe.min() - margin_hz, fringe.max() + margin_hz
        rate_profile = _roll_off(
            _outside(rates_hz, low_hz, high_hz, rate_half_period), rate_roll_off_hz
        )
        removal = np.maximum(removal, np.outer(rate_profile, delay_profile))

        # The delay notch alone, per row, as the reference for the check
        distance = (
            np.abs(_wrap(delays_s[None, :] - delay[:, None], delay_half_period))
            - delay_width_s
        )
        window_k = (distance <= delay_roll_off_s) & active[:, None]
        all_windows |= window_k
        row_notch = np.minimum(
            row_notch, 1 - _roll_off(distance, delay_roll_off_s) * active[:, None]
        )

        meets_field = (
            _outside(np.zeros(1), low_s, high_s, delay_half_period)[0]
            < field_delay_s + delay_roll_off_s
        )
        if not meets_field:
            continue
        apart = (
            low_hz - rate_roll_off_hz > field_rate_hz
            or high_hz + rate_roll_off_hz < -field_rate_hz
        )
        if not apart or max(abs(low_hz), abs(high_hz)) >= rate_half_period:
            return SegmentFilterResult(outcome=SegmentOutcome.NOT_SEPARABLE)
        checks.append((rate_profile, window_k, active))

    field_box = (np.abs(rates_hz)[:, None] <= field_rate_hz) & (
        np.abs(delays_s)[None, :] <= field_delay_s
    )
    removal[field_box] = 0.0

    weights = (~flags).astype(float) * window[:, None, None]
    spectrum = np.fft.fft2(
        np.where(flags, 0, data) * weights, s=(n_time, n_chan), axes=(0, 1)
    )
    model = np.fft.ifft2(spectrum * removal[..., None], axes=(0, 1))[:n_rows, :n_chan]
    filtered = np.where(usable[:, None, None], data - model, data)

    notched = np.fft.ifft(
        np.fft.fft(np.where(flags, 0, data), axis=1) * row_notch[..., None], axis=1
    )
    recovered = usable.copy()
    for rate_profile, window_k, active in checks:
        after, noise = _band_residual(
            filtered, weights, rate_profile, window_k, all_windows, n_time
        )
        reference, _ = _band_residual(
            notched, weights, rate_profile, window_k, all_windows, n_time
        )
        passes = after <= acceptance_factor * reference + acceptance_sigma * noise
        recovered &= np.all(passes, axis=1) | ~active

    removed = np.fft.fftshift(np.mean(removal, axis=0))
    return SegmentFilterResult(
        outcome=SegmentOutcome.FILTERED,
        data=filtered,
        recovered=recovered,
        removed_fraction=np.broadcast_to(removed, (n_rows, n_chan)),
        removal=removal,
        window=window,
    )


def segment_spectrum(
    data: NDArray[np.complexfloating[Any]],
    flags: NDArray[np.bool_],
    window: NDArray[np.floating[Any]],
    n_time: int,
) -> NDArray[np.floating[Any]]:
    """Stokes I amplitude of a windowed segment in fringe rate and delay, shape (rate, delay) in FFT order"""
    weighted = np.where(flags, 0, data) * window[:, None, None]
    spectrum = np.fft.fft2(weighted, s=(n_time, data.shape[1]), axes=(0, 1))
    return np.abs(spectrum[..., 0] + spectrum[..., -1]) / 2


def rate_separation_hz(
    w_delays: WDelays,
    baseline_idx: NDArray[np.int_],
    time_idx: NDArray[np.int_],
    freq_hz: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """The gap in fringe rate between an object and the field's guard over the
    band at each row, in Hz. Zero where they overlap, or the object is beyond
    the Nyquist rate.

    Args:
        w_delays (WDelays): The object's delays and the field's rate guard
        baseline_idx (NDArray[np.int_]): The baseline index of each row
        time_idx (NDArray[np.int_]): The time index of each row
        freq_hz (NDArray[np.floating[Any]]): The frequency of each channel, in Hz

    Returns:
        NDArray[np.floating[Any]]: The gap at each row, in Hz
    """
    rate = np.abs(get_w_rates(w_delays)[baseline_idx, time_idx].to("").value)
    guard = (
        np.zeros_like(rate)
        if w_delays.rate_guard_region is None
        else w_delays.rate_guard_region[baseline_idx, time_idx].to("").value
    )
    time_s = _time_map_to_seconds(w_delays.time_map)
    step_s = float(np.median(np.diff(time_s))) if len(time_s) > 1 else np.inf
    nu_low, nu_high = np.min(np.abs(freq_hz)), np.max(np.abs(freq_hz))
    gap = rate * nu_low - guard * nu_high
    return np.where((gap > 0) & (rate * nu_high < 0.5 / step_s), gap, 0.0)


def needed_duration_s(
    separation_hz: NDArray[np.floating[Any]],
    rate_margin_lobes: float = 4.0,
    rate_roll_off_bins: int = 2,
) -> NDArray[np.floating[Any]]:
    """The shortest segment that ``filter_segment``, with the same settings,
    can separate an object from the field over: the margin around the object,
    the roll-off and the field's main lobe all scale as 1/T

    Args:
        separation_hz (NDArray[np.floating[Any]]): The gap from ``rate_separation_hz``, in Hz
        rate_margin_lobes (float, optional): As for ``filter_segment``. Defaults to 4.0.
        rate_roll_off_bins (int, optional): As for ``filter_segment``. Defaults to 2.

    Returns:
        NDArray[np.floating[Any]]: The duration, in seconds. Infinite where the gap is zero.
    """
    lobes = rate_margin_lobes + 1 + rate_roll_off_bins / 2
    with np.errstate(divide="ignore"):
        return np.where(separation_hz > 0, lobes / separation_hz, np.inf)


@dataclass
class RowUpdate:
    """Recovered rows of one baseline to write back"""

    rows: NDArray[np.int_]
    """The row numbers"""
    data: NDArray[np.complexfloating[Any]]
    """Filtered visibilities, shape (row, chan, pol)"""
    flags: NDArray[np.bool_]
    """Flags before the delay notch, shape (row, chan, pol)"""
    weights: dict[str, NDArray[np.floating[Any]]] | None = None
    """Scaled weights, keyed by column name"""


@dataclass
class RateFilterSummary:
    """Row counts over a run of delay-rate filtering"""

    recovered: int = 0
    rejected: int = 0
    too_short: int = 0
    not_separable: int = 0
    plotted: int = 0


class SegmentFilter(Protocol):
    """Removes the objects from one baseline segment, as ``filter_segment`` does"""

    def __call__(
        self,
        *,
        data: NDArray[np.complexfloating[Any]],
        flags: NDArray[np.bool_],
        is_pad: NDArray[np.bool_],
        time_s: NDArray[np.floating[Any]],
        freq_hz: NDArray[np.floating[Any]],
        geometry: SegmentGeometry,
    ) -> SegmentFilterResult: ...


@dataclass(frozen=True)
class RateFilterSettings:
    """What delay-rate filtering needs to know of the filter and the objects"""

    w_delays_list: list[WDelays]
    """The delays of each object to null"""
    freq_hz: NDArray[np.floating[Any]]
    """Frequency of each channel, in Hz"""
    segment_filter: SegmentFilter
    """Removes the objects from each segment, e.g. ``filter_segment`` with its delay widths bound"""
    field_delay_s: float
    """Half-width of the field's delays before the guard band, in seconds"""
    elevation_cut_deg: float
    """Objects below this elevation are not nulled"""
    ignore_nyquist_zone: int
    """Objects beyond this Nyquist zone are not nulled"""
    pad_timesteps: int = 16
    """Clean rows either side of a segment"""
    plot_dir: Path | None = None
    """Where to save segment plots. None for no plots."""
    max_plots: int = 10
    """The most segments to plot"""


@dataclass
class RateFilterRun:
    """The state of delay-rate filtering over a measurement set"""

    settings: RateFilterSettings
    accumulator: SegmentAccumulator
    summary: RateFilterSummary = field(default_factory=RateFilterSummary)


def make_rate_filter_run(
    settings: RateFilterSettings,
    max_gap_steps: float = 1.5,
) -> RateFilterRun:
    """Start delay-rate filtering, feeding chunks in row order to ``update_rate_filter``

    Args:
        settings (RateFilterSettings): The notch and object settings
        max_gap_steps (float, optional): A step in time of more than this many integrations ends a segment. Defaults to 1.5.

    Returns:
        RateFilterRun: The empty state
    """
    time_s = _time_map_to_seconds(settings.w_delays_list[0].time_map)
    return RateFilterRun(
        settings=settings,
        accumulator=SegmentAccumulator(
            pad_timesteps=settings.pad_timesteps,
            max_time_gap_s=max_gap_steps * float(np.median(np.diff(time_s)))
            if len(time_s) > 1
            else None,
        ),
    )


def segment_geometry(
    segment: CandidateSegment, settings: RateFilterSettings
) -> SegmentGeometry:
    """Predicted delays and rates of the objects and the field over a segment"""
    baseline = segment.baseline_idx
    time_idx = np.asarray(segment.time_idx)
    half_period_s = 0.5 / abs(float(np.mean(np.diff(settings.freq_hz))))
    delays, rates, active = [], [], []
    for w_delays in settings.w_delays_list:
        delay = w_delays.w_delays[baseline, time_idx].to("s").value
        elevation = w_delays.elevation[time_idx].to("deg").value
        zone = calculate_nyquist_zone(values=delay, upper_limit=half_period_s)
        delays.append(delay)
        rates.append(get_w_rates(w_delays)[baseline, time_idx].to("").value)
        active.append(
            (elevation >= settings.elevation_cut_deg)
            & (zone <= settings.ignore_nyquist_zone)
        )
    first = settings.w_delays_list[0]
    no_guard = np.zeros(len(time_idx))
    return SegmentGeometry(
        object_names=tuple(w.object_name for w in settings.w_delays_list),
        object_delay_s=np.array(delays),
        object_rate=np.array(rates),
        object_active=np.array(active),
        field_delay_s=settings.field_delay_s
        + (
            no_guard
            if first.guard_region is None
            else first.guard_region[baseline, time_idx].to("s").value
        ),
        field_rate=no_guard
        if first.rate_guard_region is None
        else first.rate_guard_region[baseline, time_idx].to("").value,
    )


def filter_candidate_segment(
    segment: CandidateSegment,
    settings: RateFilterSettings,
    summary: RateFilterSummary,
) -> RowUpdate | None:
    """Filter one segment, returning the rows it recovers, if any

    Args:
        segment (CandidateSegment): The candidates and pad rows of one baseline
        settings (RateFilterSettings): The notch and object settings
        summary (RateFilterSummary): Row counts, updated in place

    Returns:
        RowUpdate | None: The recovered rows, or None if there are none
    """
    n_candidates = segment.n_candidates
    geometry = segment_geometry(segment, settings)
    result = settings.segment_filter(
        data=np.array(segment.data),
        flags=np.array(segment.mask),
        is_pad=np.array(segment.is_pad),
        time_s=np.array(segment.time_mjds),
        freq_hz=settings.freq_hz,
        geometry=geometry,
    )
    if result.outcome == SegmentOutcome.TOO_SHORT:
        summary.too_short += n_candidates
    elif result.outcome == SegmentOutcome.NOT_SEPARABLE:
        summary.not_separable += n_candidates
    if result.outcome != SegmentOutcome.FILTERED:
        return None
    assert result.data is not None
    assert result.recovered is not None

    if settings.plot_dir is not None and summary.plotted < settings.max_plots:
        from jolly_roger.plots import plot_rate_filter_segment  # noqa: PLC0415

        plot_rate_filter_segment(
            segment=segment,
            result=result,
            geometry=geometry,
            freq_hz=settings.freq_hz,
            output_path=settings.plot_dir
            / f"rate_filter_{segment.ant_1}_{segment.ant_2}_{segment.rows[0]}.png",
        )
        summary.plotted += 1

    recovered = result.recovered
    summary.recovered += int(recovered.sum())
    summary.rejected += n_candidates - int(recovered.sum())
    if not recovered.any():
        return None

    weights = None
    if segment.weights is not None and result.removed_fraction is not None:
        weights = scale_multiple_weights(
            taper=1.0 - result.removed_fraction[recovered][..., None],
            weights={
                name: np.array(values)[recovered]
                for name, values in segment.weights.items()
            },
        )
    return RowUpdate(
        rows=np.array(segment.rows)[recovered],
        data=result.data[recovered],
        flags=np.array(segment.mask)[recovered],
        weights=weights,
    )


def _filter_segments(
    run: RateFilterRun, segments: list[CandidateSegment]
) -> list[RowUpdate]:
    updates = [
        filter_candidate_segment(segment, run.settings, run.summary)
        for segment in segments
    ]
    return [update for update in updates if update is not None]


def update_rate_filter(
    run: RateFilterRun, payload: RateFilterPayload
) -> list[RowUpdate]:
    """Add a chunk, returning the rows recovered from segments that end in it"""
    return _filter_segments(run, update_segment_accumulator(run.accumulator, payload))


def flush_rate_filter(run: RateFilterRun) -> list[RowUpdate]:
    """Filter the segments still being collected"""
    return _filter_segments(run, flush_segment_accumulator(run.accumulator))


def describe_rate_filter(run: RateFilterRun) -> str:
    """Summarise delay-rate filtering for the log and HISTORY"""
    summary = run.summary
    doubly_contaminated_rows = run.accumulator.doubly_contaminated_rows
    return (
        f"Delay-rate filter: recovered {summary.recovered} rows; "
        f"not recovered: {summary.rejected} rejected by the filter's checks, "
        f"{summary.too_short} in short segments, "
        f"{summary.not_separable + doubly_contaminated_rows} not separable in delay-rate"
    )
