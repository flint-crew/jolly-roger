"""Delay-rate filtering of the timesteps where an object is in the delay
contaminated zone, i.e. where the object can not be separated from the
field by delay alone.

Rows of a measurement set are streamed in time-ordered chunks of many baselines.
The ``SegmentAccumulator`` (via ``update_segment_accumulator``) collects, per-baseline, the rows that are
recoverable (contaminated in delay but separable in delay-rate). Once the
object leaves the contaminated zone the collected segment is released and
filtered in two dimensions (delay and delay-rate) by ``rate_filter_segment``.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from jolly_roger.delays import array_to_delay_rate, delay_rate_to_array
from jolly_roger.logging import logger
from jolly_roger.response import calculate_expected_rate_sinc_width
from jolly_roger.tapering.tukey import get_2d_taper
from jolly_roger.uvws import WDelays, get_w_rates
from jolly_roger.weights import scale_weights
from jolly_roger.wrap import calculate_nyquist_zone, symmetric_domain_wrap


@dataclass(frozen=True)
class RateFilterSettings:
    """The subset of tractor options needed to filter in delay-rate space"""

    outer_width_ns: float
    """The width of the notch beyond the object's delay track, in nanoseconds"""
    tukey_width_ns: float
    """The width of the transition region of the notch in delay, in nanoseconds. If zero ``outer_width_ns`` is used, so the taper is never a hard edge."""
    width_hz: float | None = None
    """The width beyond the object's fringe-rate band over which the taper rolls off from zero to one, in Hz. If None two rate bins are used."""
    guard_hz: float | None = None
    """A minimum fringe-rate around zero to protect, added to the geometric rate guard. If None one rate bin is used."""
    min_timesteps: int = 8
    """The minimum number of contaminated timesteps a segment needs to be filtered"""
    elevation_cut_deg: float = -1.0
    """Objects below this elevation are not nulled"""
    ignore_nyquist_zone: int = 2
    """Objects beyond this Nyquist zone in delay are not nulled"""
    auto_width: bool = False
    """Size the margin of each segment from its expected sinc response in delay-rate, overriding ``width_hz``. The margin covers ``auto_sidelobes`` sidelobes, reduced towards the main lobe as needed to keep the object separable from the field."""
    auto_sidelobes: int = 1
    """The number of delay-rate sidelobes the margin includes when ``auto_width`` is set"""


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
class ContaminatedSegment:
    """The consecutive timesteps of a baseline that are to be filtered
    in delay-rate space. Core rows are those that are contaminated. Pad
    rows are clean neighbouring timesteps that only improve the rate
    resolution and are never written back."""

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
    """Whether a row is padding rather than contaminated"""
    weights: dict[str, list[NDArray[np.floating[Any]]]] | None = None
    """The original weights of each row, keyed by column name"""

    @property
    def n_rows(self) -> int:
        """The total number of rows, core and pad"""
        return len(self.rows)

    @property
    def core(self) -> NDArray[np.bool_]:
        """Mask selecting the core (contaminated) rows"""
        return ~np.array(self.is_pad, dtype=bool)

    @property
    def n_core(self) -> int:
        """The number of core (contaminated) rows"""
        return int(np.sum(self.core))

    @property
    def core_rows(self) -> NDArray[np.int_]:
        """The measurement set row numbers of the core rows"""
        return np.array(self.rows, dtype=int)[self.core]


def _append_segment_row(segment: ContaminatedSegment, row: _Row, is_pad: bool) -> None:
    """Add a row to the end of a segment"""
    segment.rows.append(row.row)
    segment.time_mjds.append(row.time_mjd)
    segment.time_idx.append(row.time_idx)
    segment.data.append(row.data)
    segment.mask.append(row.mask)
    segment.is_pad.append(is_pad)
    if row.weights is not None:
        if segment.weights is None:
            segment.weights = {k: [] for k in row.weights}
        for k, v in row.weights.items():
            segment.weights[k].append(v)


@dataclass
class _BaselineState:
    """Tracking of a single baseline in the accumulator"""

    segment: ContaminatedSegment | None = None
    """The segment being collected. None when idle."""
    trailing: int | None = None
    """Number of trailing pad rows collected. None when the segment is still open."""
    leading: deque[_Row] = field(default_factory=deque)
    """The most recent consecutive clean rows, used to lead a new segment"""
    last_time_idx: int | None = None
    """The time index of the last row seen"""


@dataclass
class SegmentAccumulator:
    """Collect recoverable rows of each baseline into segments as the object
    enters the contaminated zone, and release a segment once the object exits.
    Rows are consumed by ``update_segment_accumulator``.

    Each row is one of:
    - clean: not contaminated in delay
    - recoverable: contaminated in delay but separable in delay-rate
    - doubly contaminated: contaminated in delay and delay-rate

    A segment is released when a baseline sees a clean row (after ``pad_timesteps``
    trailing clean rows have been collected), a doubly contaminated row, a gap
    in time, or ``max_timesteps`` core rows. Up to ``pad_timesteps`` clean rows
    preceding a segment are prepended as padding.
    """

    max_timesteps: int | None = None
    """The maximum number of core rows of a segment before it is released. If None there is no limit."""
    pad_timesteps: int = 0
    """The number of clean rows either side of a segment to include as padding"""
    states: dict[tuple[int, int], _BaselineState] = field(default_factory=dict)
    """The tracking of each baseline, keyed by (ANTENNA1, ANTENNA2)"""
    doubly_contaminated_rows: int = 0
    """Count of rows that were contaminated in both delay and delay-rate"""


def _release_segment(
    state: _BaselineState, released: list[ContaminatedSegment]
) -> None:
    """Move the segment of a baseline, if any, to the released segments"""
    if state.segment is not None:
        released.append(state.segment)
    state.segment = None
    state.trailing = None


def update_segment_accumulator(
    accumulator: SegmentAccumulator,
    row_numbers: NDArray[np.int_],
    ant_1: NDArray[np.int_],
    ant_2: NDArray[np.int_],
    baseline_idx: NDArray[np.int_],
    time_mjds: NDArray[np.floating[Any]],
    time_idx: NDArray[np.int_],
    data: NDArray[np.complexfloating[Any]],
    mask: NDArray[np.bool_],
    recoverable: NDArray[np.bool_],
    delay_contaminated: NDArray[np.bool_],
    weights: dict[str, NDArray[np.floating[Any]]] | None = None,
) -> list[ContaminatedSegment]:
    """Consume a set of rows, in the order they appear in the measurement set.

    Args:
        accumulator (SegmentAccumulator): The per-baseline collection of rows, updated in place
        row_numbers (NDArray[np.int_]): The row number of each row in the measurement set
        ant_1 (NDArray[np.int_]): The first antenna of each row
        ant_2 (NDArray[np.int_]): The second antenna of each row
        baseline_idx (NDArray[np.int_]): The index of each row's baseline into the ``WDelays``
        time_mjds (NDArray[np.floating[Any]]): The time of each row, MJD in seconds
        time_idx (NDArray[np.int_]): The index of each row's time into the ``WDelays``
        data (NDArray[np.complexfloating[Any]]): The original visibilities, shape (row, chan, pol)
        mask (NDArray[np.bool_]): The original flags, shape (row, chan, pol)
        recoverable (NDArray[np.bool_]): Rows contaminated in delay that are separable in delay-rate
        delay_contaminated (NDArray[np.bool_]): Rows contaminated in delay
        weights (dict[str, NDArray[np.floating[Any]]] | None, optional): The original weights of each row. Defaults to None.

    Returns:
        list[ContaminatedSegment]: Segments that have been released and are ready to filter
    """
    pad = accumulator.pad_timesteps
    max_timesteps = accumulator.max_timesteps
    assert pad >= 0, f"{pad=}, should be non-negative"
    assert max_timesteps is None or max_timesteps > 0, (
        f"{max_timesteps=}, should be positive"
    )

    released: list[ContaminatedSegment] = []
    for i in range(len(row_numbers)):
        is_recoverable = bool(recoverable[i])
        is_clean = not bool(delay_contaminated[i])
        key = (int(ant_1[i]), int(ant_2[i]))
        state = accumulator.states.get(key)

        if state is None:
            # Nothing is needed of idle baselines that see clean rows
            if is_clean and pad == 0:
                continue
            state = accumulator.states.setdefault(
                key, _BaselineState(leading=deque(maxlen=pad))
            )

        t_idx = int(time_idx[i])
        if state.last_time_idx is not None and t_idx - state.last_time_idx > 1:
            _release_segment(state, released)
            state.leading.clear()
        state.last_time_idx = t_idx

        if not is_clean and not is_recoverable:
            accumulator.doubly_contaminated_rows += 1
            _release_segment(state, released)
            state.leading.clear()
            continue

        # Copy so the chunk the row was drawn from can be released
        row = _Row(
            row=int(row_numbers[i]),
            time_mjd=float(time_mjds[i]),
            time_idx=t_idx,
            data=np.array(data[i]),
            mask=np.array(mask[i]),
            weights=None
            if weights is None
            else {k: np.array(v[i]) for k, v in weights.items()},
        )

        if is_recoverable:
            if state.trailing is not None:
                _release_segment(state, released)
            if state.segment is None:
                state.segment = ContaminatedSegment(
                    ant_1=key[0], ant_2=key[1], baseline_idx=int(baseline_idx[i])
                )
                for lead in state.leading:
                    _append_segment_row(state.segment, lead, is_pad=True)
            state.leading.clear()
            _append_segment_row(state.segment, row, is_pad=False)

            if max_timesteps is not None and state.segment.n_core >= max_timesteps:
                _release_segment(state, released)
            continue

        # Clean row
        if state.segment is not None:
            if pad == 0:
                _release_segment(state, released)
            else:
                _append_segment_row(state.segment, row, is_pad=True)
                state.trailing = (state.trailing or 0) + 1
                if state.trailing >= pad:
                    _release_segment(state, released)
        if pad > 0:
            state.leading.append(row)

    return released


def flush_segment_accumulator(
    accumulator: SegmentAccumulator,
) -> list[ContaminatedSegment]:
    """Release all segments still being collected, e.g. at the end of the measurement set

    Args:
        accumulator (SegmentAccumulator): The per-baseline collection of rows, emptied in place

    Returns:
        list[ContaminatedSegment]: The remaining segments
    """
    released: list[ContaminatedSegment] = []
    for state in accumulator.states.values():
        _release_segment(state, released)
    accumulator.states = {}
    return released


@dataclass(frozen=True)
class RateBox:
    """A symmetric region in delay and fringe-rate"""

    delay_center_s: float
    delay_half_width_s: float
    rate_center_hz: float
    rate_half_width_hz: float


@dataclass
class ObjectTrack:
    """The predicted path of an object through delay and fringe-rate across a segment"""

    object_name: str
    """The name of the object"""
    delay_s: NDArray[np.floating[Any]]
    """The predicted delay of each row, in seconds"""
    rate_hz: NDArray[np.floating[Any]]
    """The predicted fringe-rate of each row at the central frequency, in Hz"""
    notch: RateBox
    """The region nulled for the object"""
    rate_width_hz: float
    """The margin beyond the object's fringe-rate band over which the taper rolls off, in Hz"""


@dataclass
class RateFilterDiagnostics:
    """Quantities describing how a segment was filtered, used for plotting"""

    ant_1: int
    """The first antenna of the baseline"""
    ant_2: int
    """The second antenna of the baseline"""
    first_row: int
    """The first row of the segment in the measurement set"""
    delay_s: NDArray[np.floating[Any]]
    """The delay axis, in seconds"""
    rate_hz: NDArray[np.floating[Any]]
    """The fringe-rate axis, in Hz"""
    before: NDArray[np.floating[Any]]
    """The amplitude before filtering, averaged over polarisations. shape=(rate, delay)"""
    after: NDArray[np.floating[Any]]
    """The amplitude after filtering, averaged over polarisations. shape=(rate, delay)"""
    field: RateBox
    """The region occupied by the field, which is not modified"""
    tracks: list[ObjectTrack]
    """The predicted path of each nulled object"""


@dataclass
class RateFilterResult:
    """The outcome of filtering a segment in delay-rate space"""

    rows: NDArray[np.int_]
    """The measurement set rows of the core of the segment"""
    success: bool
    """Whether the segment was filtered"""
    reason: str = "filtered"
    """Description of the outcome"""
    data: NDArray[np.complexfloating[Any]] | None = None
    """The filtered visibilities of the core rows"""
    flags: NDArray[np.bool_] | None = None
    """The flags of the core rows, excluding those set due to contamination"""
    weights: dict[str, NDArray[np.floating[Any]]] | None = None
    """The scaled weights of the core rows"""
    nulled_fraction: float = 0.0
    """The fraction of the delay-rate plane that was nulled"""
    diagnostics: RateFilterDiagnostics | None = None
    """Description of the filtering for plotting. Only set when requested and the segment was filtered."""
    notches: list[tuple[str, RateBox]] = field(default_factory=list)
    """The (object name, region) nulled for each object. Only set when the segment was filtered."""


def _wrapped_overlap(
    center_a: float, half_a: float, center_b: float, half_b: float, upper: float
) -> bool:
    """Whether two intervals overlap on a symmetric cyclic domain"""
    if half_a + half_b >= upper:
        return True
    separation = symmetric_domain_wrap(
        values=np.array([center_a - center_b]), upper_limit=upper
    )[0]
    return bool(np.abs(separation) < half_a + half_b)


def _transition_width(
    x: NDArray[np.floating[Any]], requested: float, fallback: float
) -> float:
    """The width of a taper's ``1 - cos`` transition. A non-positive ``requested``
    width uses ``fallback``, and the width is never narrower than two samples of ``x``
    so the transition is always resolved."""
    width = requested if requested > 0.0 else fallback
    return max(width, 2 * float(np.max(np.abs(np.diff(x)))))


def _axis_notch(
    x: NDArray[np.floating[Any]], center: float, outer_width: float, tukey_width: float
) -> NDArray[np.floating[Any]]:
    """A one-dimensional notch: zero within ``outer_width - tukey_width`` of ``center``,
    rising to one at ``outer_width`` with a ``1 - cos`` transition, as the delay taper"""
    return get_2d_taper(
        x=x,
        outer_width=max(outer_width, tukey_width),
        tukey_width=tukey_width,
        tukey_offset=np.array([center]),
    )[:, 0]


def _axis_protection(
    x: NDArray[np.floating[Any]], half_width: float, tukey_width: float
) -> NDArray[np.floating[Any]]:
    """A one-dimensional window that is one within ``half_width`` of zero, falling to
    zero at ``half_width + tukey_width`` with a ``1 - cos`` transition"""
    return (
        1.0
        - get_2d_taper(
            x=x, outer_width=half_width + tukey_width, tukey_width=tukey_width
        )[:, 0]
    )


def _object_box(
    tau_s: NDArray[np.floating[Any]],
    rate_extremes_hz: NDArray[np.floating[Any]],
    outer_width_s: float,
    width_hz: float,
) -> RateBox:
    """The region to null for an object: its delay track and fringe-rate band
    across a segment, widened by ``outer_width_s`` in delay and ``width_hz`` in rate"""
    return RateBox(
        delay_center_s=float(np.max(tau_s) + np.min(tau_s)) / 2,
        delay_half_width_s=float(np.max(tau_s) - np.min(tau_s)) / 2 + outer_width_s,
        rate_center_hz=float(np.max(rate_extremes_hz) + np.min(rate_extremes_hz)) / 2,
        rate_half_width_hz=float(np.max(rate_extremes_hz) - np.min(rate_extremes_hz))
        / 2
        + width_hz,
    )


def _boxes_overlap(
    box_a: RateBox, box_b: RateBox, max_delay_s: float, max_rate_hz: float
) -> bool:
    """Whether two regions intersect in both delay and fringe-rate"""
    return _wrapped_overlap(
        box_a.delay_center_s,
        box_a.delay_half_width_s,
        box_b.delay_center_s,
        box_b.delay_half_width_s,
        max_delay_s,
    ) and _wrapped_overlap(
        box_a.rate_center_hz,
        box_a.rate_half_width_hz,
        box_b.rate_center_hz,
        box_b.rate_half_width_hz,
        max_rate_hz,
    )


def _candidate_rate_widths(
    settings: RateFilterSettings, time_s: NDArray[np.floating[Any]], rate_bin_hz: float
) -> list[float]:
    """The margins beyond the object's fringe-rate band to try, widest first. With
    ``auto_width`` these step from the main lobe plus ``auto_sidelobes`` sidelobes
    down to the main lobe of the expected sinc response in delay-rate."""
    if settings.auto_width:
        sinc_width_hz = calculate_expected_rate_sinc_width(time_s).to(u.Hz).value
        return [
            (n + 1) * sinc_width_hz
            for n in range(max(settings.auto_sidelobes, 0), -1, -1)
        ]
    return [settings.width_hz if settings.width_hz is not None else 2 * rate_bin_hz]


def rate_filter_segment(
    segment: ContaminatedSegment,
    freq_chan: u.Quantity,
    w_delays_list: list[WDelays],
    settings: RateFilterSettings,
    keep_diagnostics: bool = False,
) -> RateFilterResult:
    """Null the objects in ``w_delays_list`` from a segment in delay and
    delay-rate space while protecting the field.

    The object is nulled over a box spanning its predicted delay track and
    fringe-rate band across the segment. The field occupies a box around
    (delay, rate) = (0, 0) whose size is set by the guard regions. Should
    the two boxes intersect the object can not be separated and the segment
    is not filtered.

    Args:
        segment (ContaminatedSegment): The rows to filter
        freq_chan (u.Quantity): The frequency of each channel
        w_delays_list (list[WDelays]): The objects to null
        settings (RateFilterSettings): Parameterisation of the filter
        keep_diagnostics (bool, optional): Attach the quantities needed to plot the filtering. Defaults to False.

    Returns:
        RateFilterResult: The filtered core rows, or the reason they could not be filtered
    """
    core = segment.core
    core_rows = segment.core_rows
    if segment.n_core < settings.min_timesteps or segment.n_rows < 2:
        return RateFilterResult(rows=core_rows, success=False, reason="too short")

    time_s = np.array(segment.time_mjds)
    masked_data = np.ma.masked_array(
        np.array(segment.data), mask=np.array(segment.mask)
    )
    delay_rate = array_to_delay_rate(
        masked_data=masked_data, freq_chan=freq_chan, time_s=time_s
    )
    delay_s = delay_rate.delay.to(u.s).value
    rate_hz = delay_rate.rate.to(u.Hz).value
    max_delay_s = float(np.max(np.abs(delay_s)))
    max_rate_hz = float(np.max(np.abs(rate_hz)))
    rate_bin_hz = 1.0 / (len(time_s) * float(np.mean(np.diff(time_s))))

    freq_hz = freq_chan.to(u.Hz).value
    nu_min, nu_max = float(np.min(freq_hz)), float(np.max(freq_hz))
    outer_width_s = settings.outer_width_ns * 1e-9
    tukey_width_s = settings.tukey_width_ns * 1e-9
    candidate_widths_hz = _candidate_rate_widths(
        settings=settings, time_s=time_s, rate_bin_hz=rate_bin_hz
    )
    floor_hz = settings.guard_hz if settings.guard_hz is not None else rate_bin_hz
    # The widths of the 1 - cos transitions of the taper. The field is protected
    # with the narrowest margin that may be used for an object
    delay_transition_s = _transition_width(
        x=delay_s, requested=tukey_width_s, fallback=outer_width_s
    )
    field_rate_transition_hz = _transition_width(
        x=rate_hz,
        requested=candidate_widths_hz[-1],
        fallback=candidate_widths_hz[-1],
    )

    b_idx = segment.baseline_idx
    t_idx = np.array(segment.time_idx)

    # The guard regions are common to all objects
    reference = w_delays_list[0]
    delay_guard_s = (
        float(np.max(reference.guard_region[b_idx, t_idx].to(u.s).value))
        if reference.guard_region is not None
        else 0.0
    )
    rate_guard = (
        float(np.max(reference.rate_guard_region[b_idx, t_idx].value))
        if reference.rate_guard_region is not None
        else 0.0
    )
    field = RateBox(
        delay_center_s=0.0,
        delay_half_width_s=outer_width_s + delay_guard_s,
        rate_center_hz=0.0,
        rate_half_width_hz=nu_max * rate_guard + floor_hz,
    )

    nu_mid = (nu_min + nu_max) / 2
    tracks: list[ObjectTrack] = []
    notch = np.ones((len(rate_hz), len(delay_s)))
    for w_delays in w_delays_list:
        elevation = w_delays.elevation[t_idx]
        if np.all(elevation < settings.elevation_cut_deg * u.deg):
            continue
        tau_s = w_delays.w_delays[b_idx, t_idx].to(u.s).value
        if np.all(
            calculate_nyquist_zone(values=tau_s, upper_limit=max_delay_s)
            > settings.ignore_nyquist_zone
        ):
            continue
        tau_rate = get_w_rates(w_delays)[b_idx, t_idx].value

        # The fringe-rate of the object spans the band and its change across the segment
        rate_extremes = np.outer([nu_min, nu_max], [np.min(tau_rate), np.max(tau_rate)])

        # Use the widest margin that keeps the object separable from the field
        fitted: tuple[RateBox, float] | None = None
        for width_hz in candidate_widths_hz:
            candidate = _object_box(
                tau_s=tau_s,
                rate_extremes_hz=rate_extremes,
                outer_width_s=outer_width_s,
                width_hz=width_hz,
            )
            if not _boxes_overlap(candidate, field, max_delay_s, max_rate_hz):
                fitted = (candidate, width_hz)
                break
        if fitted is None:
            return RateFilterResult(
                rows=core_rows, success=False, reason="rate-contaminated"
            )
        obj, width_hz = fitted
        if width_hz < candidate_widths_hz[0]:
            logger.debug(
                f"Reduced the delay-rate margin of {w_delays.object_name} to {width_hz:.3g} Hz "
                f"(from {candidate_widths_hz[0]:.3g} Hz) to stay separable from the field"
            )

        delay_notch = _axis_notch(
            x=delay_s,
            center=obj.delay_center_s,
            outer_width=obj.delay_half_width_s,
            tukey_width=delay_transition_s,
        )
        rate_notch = _axis_notch(
            x=rate_hz,
            center=obj.rate_center_hz,
            outer_width=obj.rate_half_width_hz,
            tukey_width=_transition_width(
                x=rate_hz, requested=width_hz, fallback=width_hz
            ),
        )
        object_notch = 1.0 - (1.0 - rate_notch[:, None]) * (1.0 - delay_notch[None, :])
        notch = np.minimum(notch, object_notch)
        tracks.append(
            ObjectTrack(
                object_name=w_delays.object_name,
                delay_s=tau_s,
                rate_hz=nu_mid * tau_rate,
                notch=obj,
                rate_width_hz=width_hz,
            )
        )

    if np.all(notch == 1.0):
        return RateFilterResult(rows=core_rows, success=False, reason="nothing to null")

    # Never modify the field. The protection is one across the field and rolls
    # off smoothly, so the taper has no discontinuity at the field's boundary
    protection = (
        _axis_protection(
            x=rate_hz,
            half_width=field.rate_half_width_hz,
            tukey_width=field_rate_transition_hz,
        )[:, None]
        * _axis_protection(
            x=delay_s,
            half_width=field.delay_half_width_s,
            tukey_width=delay_transition_s,
        )[None, :]
    )
    notch = 1.0 - (1.0 - notch) * (1.0 - protection)

    before = np.abs(delay_rate.delay_rate).mean(axis=-1) if keep_diagnostics else None
    delay_rate.delay_rate = delay_rate.delay_rate * notch[..., None]
    diagnostics: RateFilterDiagnostics | None = None
    if before is not None:
        diagnostics = RateFilterDiagnostics(
            ant_1=segment.ant_1,
            ant_2=segment.ant_2,
            first_row=int(segment.rows[0]),
            delay_s=delay_s,
            rate_hz=rate_hz,
            before=before,
            after=np.abs(delay_rate.delay_rate).mean(axis=-1),
            field=field,
            tracks=tracks,
        )
    filtered = delay_rate_to_array(delay_rate)[core]

    original_mask = np.array(segment.mask)[core]
    flags = original_mask | ~np.isfinite(filtered)

    mean_notch = float(np.mean(notch))
    scaled_weights: dict[str, NDArray[np.floating[Any]]] | None = None
    if segment.weights is not None:
        scale = np.full(segment.n_core, 1.0 / max(mean_notch, 1e-12))
        scaled_weights = {
            k: scale_weights(
                taper=scale, weights=np.array(v)[core], taper_is_scale=True
            )
            for k, v in segment.weights.items()
        }

    return RateFilterResult(
        rows=core_rows,
        success=True,
        data=filtered,
        flags=flags,
        weights=scaled_weights,
        nulled_fraction=1.0 - mean_notch,
        diagnostics=diagnostics,
        notches=[(track.object_name, track.notch) for track in tracks],
    )


@dataclass
class RateFilterSummary:
    """Tally of the delay-rate filtering outcomes"""

    segments_filtered: int = 0
    """The number of segments that were filtered"""
    rows_filtered: int = 0
    """The number of core rows that were filtered"""
    failures: Counter[str] = field(default_factory=Counter)
    """The number of segments not filtered, by reason"""
    rows_not_filtered: int = 0
    """The number of core rows that were not filtered"""


def record_rate_filter_result(
    summary: RateFilterSummary, result: RateFilterResult
) -> None:
    """Add the outcome of filtering a segment to the summary

    Args:
        summary (RateFilterSummary): The tally, updated in place
        result (RateFilterResult): The outcome of a segment
    """
    if result.success:
        summary.segments_filtered += 1
        summary.rows_filtered += len(result.rows)
    else:
        summary.failures[result.reason] += 1
        summary.rows_not_filtered += len(result.rows)


def log_rate_filter_summary(
    summary: RateFilterSummary, doubly_contaminated_rows: int = 0
) -> None:
    """Log the outcomes of the delay-rate filtering

    Args:
        summary (RateFilterSummary): The tally to log
        doubly_contaminated_rows (int, optional): Rows contaminated in delay and delay-rate. Defaults to 0.
    """
    logger.info(
        f"Delay-rate filtered {summary.segments_filtered} segments ({summary.rows_filtered} rows)"
    )
    if summary.failures:
        logger.info(
            f"Segments not filtered ({summary.rows_not_filtered} rows): {dict(summary.failures)}"
        )
    logger.info(
        f"{doubly_contaminated_rows} rows were contaminated in delay and delay-rate"
    )
