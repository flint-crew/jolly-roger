"""Delay-rate filtering of the timesteps where an object is in the delay
contaminated zone, i.e. where the object can not be separated from the
field by delay alone.

Rows of a measurement set are streamed in time-ordered chunks of many baselines.
``update_segment_accumulator`` collects, per-baseline, the rows that are
recoverable: contaminated in delay, but where every object is separable from the
field in delay-rate. Once the object leaves the contaminated zone the collected
segment is released and filtered in two dimensions (delay and delay-rate) by
``rate_filter_segment``. A row is only unflagged when every object passes both
the delay and the delay-rate filtering: timesteps of a segment where any object
meets the field in delay-rate remain flagged, and the rest of the segment is split
around them and filtered again (``rate_filter_segments``).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from jolly_roger.delays import array_to_delay_rate, delay_rate_to_array
from jolly_roger.logging import logger
from jolly_roger.response import (
    calculate_expected_rate_sinc_width,
    get_delay_of_nth_sidelobe,
)
from jolly_roger.tapering.tukey import get_2d_taper
from jolly_roger.uvws import WDelays, get_w_rates
from jolly_roger.weights import scale_weights
from jolly_roger.wrap import (
    axis_half_period,
    calculate_nyquist_zone,
    symmetric_domain_wrap,
)


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
    ignore_rate_nyquist_zone: int | None = 2
    """Objects are not separated in delay-rate beyond this Nyquist zone in fringe-rate. Rows where such an object is contaminated in delay remain flagged, and elsewhere it is nulled in delay only. If None there is no limit."""
    nth_sidelobe_null: int | None = None
    """Null each object's delay sidelobes up to this order, as the delay taper does. If None only the main lobe is nulled."""
    peak_shift_search: bool = False
    """Search for a shift of each object's delay, as imparted by the ionosphere, and move its notch by it"""
    peak_shift_search_width_ns: float | None = None
    """The largest delay shift searched for, in nanoseconds. If None ``outer_width_ns`` is used."""
    integration_s: float | None = None
    """The integration time of the measurement set, in seconds. Needed to classify rows before any segment is formed."""


class RateFilterOutcome(Enum):
    """The outcome of delay-rate filtering a segment. Only a filtered segment is
    written back and unflagged."""

    FILTERED = "filtered"
    TOO_SHORT = "too short"
    RATE_CONTAMINATED = "rate-contaminated"
    NOTHING_TO_NULL = "nothing to null"


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
    n_core: int = 0
    """The number of core (contaminated) rows"""
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
    if not is_pad:
        segment.n_core += 1
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
    last_time_mjd: float | None = None
    """The time of the last row seen, MJD in seconds"""


@dataclass
class SegmentAccumulator:
    """Collect recoverable rows of each baseline into segments as the object
    enters the contaminated zone, and release a segment once the object exits.
    Rows are consumed by ``update_segment_accumulator``.

    Each row is one of:
    - clean: not contaminated in delay
    - recoverable: contaminated in delay, and every object is separable in delay-rate
    - doubly contaminated: contaminated in delay, and some object is not separable in delay-rate

    A segment is released when a baseline sees a clean row (after ``pad_timesteps``
    trailing clean rows have been collected), a doubly contaminated row, a gap
    in time, or ``max_timesteps`` core rows. Up to ``pad_timesteps`` clean rows
    preceding a segment are prepended as padding.
    """

    max_timesteps: int | None = None
    """The maximum number of core rows of a segment before it is released. If None there is no limit."""
    pad_timesteps: int = 0
    """The number of clean rows either side of a segment to include as padding"""
    max_time_gap_s: float | None = None
    """The largest time between consecutive rows of a baseline within a segment, in seconds. If None a gap is any jump of more than one in time index."""
    states: dict[tuple[int, int], _BaselineState] = field(default_factory=dict)
    """The tracking of each baseline, keyed by (ANTENNA1, ANTENNA2)"""
    doubly_contaminated_rows: int = 0
    """Count of rows that were contaminated in both delay and delay-rate"""


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
    """Rows contaminated in delay by any object"""
    recoverable: NDArray[np.bool_]
    """Rows contaminated in delay where every contaminating object is separable in delay-rate"""
    original_weights: dict[str, NDArray[np.floating[Any]]] | None = None
    """The weights before any scaling"""


def _release_segment(
    state: _BaselineState, released: list[ContaminatedSegment]
) -> None:
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
    if max_time_gap_s is None:
        if state.last_time_idx is None:
            return False
        return not 0 < time_idx - state.last_time_idx <= 1
    if state.last_time_mjd is None:
        return False
    return not 0 < time_mjd - state.last_time_mjd <= max_time_gap_s


def update_segment_accumulator(
    accumulator: SegmentAccumulator, payload: RateFilterPayload
) -> list[ContaminatedSegment]:
    """Consume the rows of a chunk, in the order they appear in the measurement set.

    Args:
        accumulator (SegmentAccumulator): The per-baseline collection of rows, updated in place
        payload (RateFilterPayload): The rows, and whether each is contaminated or recoverable

    Returns:
        list[ContaminatedSegment]: Segments that have been released and are ready to filter
    """
    pad = accumulator.pad_timesteps
    max_timesteps = accumulator.max_timesteps
    data = np.ma.getdata(payload.original_masked_data)
    mask = np.ma.getmaskarray(payload.original_masked_data)
    weights = payload.original_weights

    released: list[ContaminatedSegment] = []
    for i in range(len(payload.rows)):
        is_recoverable = bool(payload.recoverable[i])
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

        if not is_clean and not is_recoverable:
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

        if is_recoverable:
            if state.trailing is not None:
                _release_segment(state, released)
            if state.segment is None:
                state.segment = ContaminatedSegment(
                    ant_1=key[0],
                    ant_2=key[1],
                    baseline_idx=int(payload.baseline_idx[i]),
                )
                for lead in state.leading:
                    _append_segment_row(state.segment, lead, is_pad=True)
            state.leading.clear()
            _append_segment_row(state.segment, row, is_pad=False)

            if max_timesteps is not None and state.segment.n_core >= max_timesteps:
                _release_segment(state, released)
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
    """The centre of the region in delay, in seconds"""
    delay_half_width_s: float
    """Half of the width of the region in delay, in seconds"""
    rate_center_hz: float
    """The centre of the region in fringe-rate, in Hz"""
    rate_half_width_hz: float
    """Half of the width of the region in fringe-rate, in Hz"""


@dataclass(frozen=True)
class RateFootprint:
    """The region of delay and fringe-rate an object occupies across a segment. At
    each time the object is at a delay, and spans a range of fringe-rates across the
    band. The region nulled is this footprint widened by a margin in delay and in
    fringe-rate."""

    object_name: str
    """The name of the object"""
    delay_s: NDArray[np.floating[Any]]
    """The delay of the object at each time, in seconds"""
    delay_occupied_half_s: NDArray[np.floating[Any]]
    """Half of the object's movement in delay between times, in seconds. The delay within this of the object is occupied."""
    delay_margin_s: float
    """The margin beyond the occupied delays over which the taper rolls off, in seconds"""
    rate_low_hz: NDArray[np.floating[Any]]
    """The fringe-rate of the object at the lowest frequency at each time, in Hz"""
    rate_high_hz: NDArray[np.floating[Any]]
    """The fringe-rate of the object at the highest frequency at each time, in Hz"""
    rate_margin_hz: float
    """The margin beyond the object's fringe-rates over which the taper rolls off, in Hz"""
    active: NDArray[np.bool_]
    """The times the object is nulled, as for the delay taper (above the elevation cut and within the delay Nyquist zone)"""
    all_rates: NDArray[np.bool_]
    """The times the object is nulled across all fringe-rates (in delay only), as it is beyond the fringe-rate Nyquist zone"""
    sidelobe: int = 0
    """The order of the delay sidelobe, signed by its side. Zero is the main lobe."""


def footprint_rate_band(
    footprint: RateFootprint, max_rate_hz: float
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """The centre and half-width of the fringe-rates a footprint occupies at each
    time, without its margin. Times nulled across all fringe-rates span more than
    the whole fringe-rate axis.

    Args:
        footprint (RateFootprint): The footprint of an object
        max_rate_hz (float): The half-period of the fringe-rate axis, in Hz

    Returns:
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]: The centre and half-width at each time, in Hz
    """
    center = (footprint.rate_low_hz + footprint.rate_high_hz) / 2
    half = np.abs(footprint.rate_high_hz - footprint.rate_low_hz) / 2
    return (
        np.where(footprint.all_rates, 0.0, center),
        np.where(footprint.all_rates, 2 * max_rate_hz, half),
    )


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
    footprints: list[RateFootprint]
    """The region nulled for each object, and each of its sidelobes"""
    taper: NDArray[np.floating[Any]]
    """The taper applied, between zero and one. shape=(rate, delay)"""


@dataclass
class RateFilterResult:
    """The outcome of filtering a segment in delay-rate space"""

    rows: NDArray[np.int_]
    """The measurement set rows of the core of the segment"""
    outcome: RateFilterOutcome
    """Whether the segment was filtered, or why not"""
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
    notches: list[RateFootprint] = field(default_factory=list)
    """The region nulled for each object, and each of its sidelobes. Only set when the segment was filtered."""
    contaminated_timesteps: NDArray[np.bool_] | None = None
    """The timesteps of the segment (core and pad) where an object meets the field. Only set when rate-contaminated."""


def _cyclic_overlap(
    offset: NDArray[np.floating[Any]],
    half_width: NDArray[np.floating[Any]],
    field_half_width: float | NDArray[np.floating[Any]],
    upper: float,
) -> NDArray[np.bool_]:
    """Whether intervals at ``offset`` from the field overlap the field's interval,
    on a symmetric cyclic domain wrapping at ``upper``. Touching intervals overlap,
    so an object at the field's centre always does."""
    reach = np.asarray(half_width) + field_half_width
    separation = np.abs(
        symmetric_domain_wrap(values=np.asarray(offset, dtype=float), upper_limit=upper)
    )
    return np.asarray((reach >= upper) | (separation <= reach))


def _beyond_rate_nyquist_zone(
    tau_rate: NDArray[np.floating[Any]],
    nu_min_hz: float,
    max_rate_hz: float,
    nyquist_zone: int | None,
) -> NDArray[np.bool_]:
    """Whether an object's lowest fringe-rate across the band is beyond a Nyquist zone"""
    if nyquist_zone is None:
        return np.zeros(np.shape(tau_rate), dtype=bool)
    return np.asarray(
        calculate_nyquist_zone(
            values=nu_min_hz * np.abs(tau_rate), upper_limit=max_rate_hz
        )
        > nyquist_zone
    )


def _rate_floor_hz(settings: RateFilterSettings, rate_bin_hz: float) -> float:
    """The fringe-rate around zero protected in addition to the geometric guard"""
    return settings.guard_hz if settings.guard_hz is not None else rate_bin_hz


def _candidate_rate_widths(
    settings: RateFilterSettings, sinc_width_hz: float, rate_bin_hz: float
) -> list[float]:
    """The margins beyond the object's fringe-rate band to try, widest first. With
    ``auto_width`` these step from the main lobe plus ``auto_sidelobes`` sidelobes
    down to the main lobe of the expected sinc response in delay-rate."""
    if settings.auto_width:
        return [
            (n + 1) * sinc_width_hz
            for n in range(max(settings.auto_sidelobes, 0), -1, -1)
        ]
    return [settings.width_hz if settings.width_hz is not None else 2 * rate_bin_hz]


def compute_rate_contamination(
    w_delays: WDelays,
    baseline_idx: NDArray[np.int_],
    time_idx: NDArray[np.int_],
    freq_chan: u.Quantity,
    settings: RateFilterSettings,
) -> NDArray[np.bool_]:
    """Identify rows where the object can not be separated from the field in
    delay-rate, before any segment is formed.

    This is the test applied to each time of a segment, without the margins set by
    the fringe-rate resolution, which is only known once a segment is formed.
    ``rate_filter_segments`` applies the full test, and leaves flagged any timestep
    that fails it. Rows where the object is beyond the fringe-rate Nyquist zone can
    not be separated.

    Args:
        w_delays (WDelays): The object delays and rates
        baseline_idx (NDArray[np.int_]): The baseline index of each row
        time_idx (NDArray[np.int_]): The time index of each row
        freq_chan (u.Quantity): The frequency of each channel
        settings (RateFilterSettings): Parameterisation of the filter, including the integration time

    Raises:
        ValueError: Raised if ``settings.integration_s`` is not set

    Returns:
        NDArray[np.bool_]: Rows contaminated in delay-rate
    """
    if settings.integration_s is None:
        msg = "The integration time (settings.integration_s) is needed to classify rows"
        raise ValueError(msg)

    tau_rate = get_w_rates(w_delays)[baseline_idx, time_idx].value
    rate_guard = (
        w_delays.rate_guard_region[baseline_idx, time_idx].value
        if w_delays.rate_guard_region is not None
        else np.zeros_like(tau_rate)
    )
    freq_hz = freq_chan.to(u.Hz).value
    nu_min, nu_max = float(np.min(freq_hz)), float(np.max(freq_hz))
    # The fringe-rate axis of any segment wraps at +/- 1 / (2 integration)
    max_rate_hz = 1.0 / (2 * settings.integration_s)
    margin_hz = 0.0 if settings.auto_width else (settings.width_hz or 0.0)

    contaminated = _cyclic_overlap(
        offset=(nu_min + nu_max) / 2 * tau_rate,
        half_width=(nu_max - nu_min) / 2 * np.abs(tau_rate) + margin_hz,
        field_half_width=nu_max * rate_guard + (settings.guard_hz or 0.0),
        upper=max_rate_hz,
    )
    return contaminated | _beyond_rate_nyquist_zone(
        tau_rate, nu_min, max_rate_hz, settings.ignore_rate_nyquist_zone
    )


def _transition_width(
    x: NDArray[np.floating[Any]], requested: float, fallback: float
) -> float:
    """The width of a taper's ``1 - cos`` transition. A non-positive ``requested``
    width uses ``fallback``, and the width is never narrower than two samples of ``x``
    so the transition is always resolved."""
    width = requested if requested > 0.0 else fallback
    return max(width, 2 * float(np.max(np.abs(np.diff(x)))))


def _axis_protection(
    x: NDArray[np.floating[Any]], half_width: float, tukey_width: float
) -> NDArray[np.floating[Any]]:
    """A one-dimensional window that is one within ``half_width`` of zero, falling to
    zero at ``half_width + tukey_width`` with a ``1 - cos`` transition"""
    return (
        1.0
        - get_2d_taper(
            x=x,
            outer_width=half_width + tukey_width,
            tukey_width=tukey_width,
            upper_limit=axis_half_period(x),
        )[:, 0]
    )


@dataclass(frozen=True)
class _SegmentGrid:
    """The delay-rate plane of a segment"""

    delay_s: NDArray[np.floating[Any]]
    """The delay axis, in seconds"""
    rate_hz: NDArray[np.floating[Any]]
    """The fringe-rate axis, in Hz"""
    time_s: NDArray[np.floating[Any]]
    """The time of each row of the segment, in seconds"""
    max_delay_s: float
    """The half-period of the delay axis, in seconds"""
    max_rate_hz: float
    """The half-period of the fringe-rate axis, in Hz"""
    nu_min_hz: float
    """The lowest frequency, in Hz"""
    nu_max_hz: float
    """The highest frequency, in Hz"""
    delay_transition_s: float
    """The width of the transition of the taper in delay, in seconds"""


def _rate_transition_hz(footprint: RateFootprint, grid: _SegmentGrid) -> float:
    """The width of the transition of a footprint's taper in fringe-rate"""
    return _transition_width(
        x=grid.rate_hz,
        requested=footprint.rate_margin_hz,
        fallback=footprint.rate_margin_hz,
    )


def _sidelobe_offsets_s(settings: RateFilterSettings) -> list[tuple[int, float]]:
    """The (signed order, delay offset) of each delay sidelobe to null, as the
    delay taper places them"""
    if not settings.nth_sidelobe_null:
        return []
    sinc_width = settings.outer_width_ns * 1e-9 * u.s
    return [
        (
            sign * order,
            sign
            * get_delay_of_nth_sidelobe(n=order, sinc_width=sinc_width).to(u.s).value,
        )
        for sign in (1, -1)
        for order in range(1, settings.nth_sidelobe_null + 1)
    ]


def _make_footprints(
    object_name: str,
    tau_s: NDArray[np.floating[Any]],
    tau_rate: NDArray[np.floating[Any]],
    active: NDArray[np.bool_],
    all_rates: NDArray[np.bool_],
    grid: _SegmentGrid,
    settings: RateFilterSettings,
    rate_margin_hz: float,
) -> list[RateFootprint]:
    """The footprints of an object's main lobe and of the delay sidelobes to null"""
    step_s = (
        np.abs(tau_rate) * np.abs(np.gradient(grid.time_s))
        if len(grid.time_s) > 1
        else np.zeros_like(tau_s)
    )
    main_lobe = RateFootprint(
        object_name=object_name,
        delay_s=tau_s,
        delay_occupied_half_s=step_s / 2,
        delay_margin_s=settings.outer_width_ns * 1e-9,
        rate_low_hz=grid.nu_min_hz * tau_rate,
        rate_high_hz=grid.nu_max_hz * tau_rate,
        rate_margin_hz=rate_margin_hz,
        active=active,
        all_rates=all_rates,
    )
    return [main_lobe] + [
        replace(main_lobe, delay_s=tau_s + offset_s, sidelobe=order)
        for order, offset_s in _sidelobe_offsets_s(settings)
    ]


def _footprint_field_overlap(
    footprint: RateFootprint, field: RateBox, grid: _SegmentGrid
) -> NDArray[np.bool_]:
    """The times at which the region nulled for an object meets the field. This
    extends as far as ``_footprint_notch`` does, which is beyond the margin should
    the transition be wider, so the field's protection never reaches the region the
    object occupies."""
    rate_center, rate_half = footprint_rate_band(footprint, grid.max_rate_hz)
    delay_overlap = _cyclic_overlap(
        offset=footprint.delay_s - field.delay_center_s,
        half_width=footprint.delay_occupied_half_s
        + max(footprint.delay_margin_s, grid.delay_transition_s),
        field_half_width=field.delay_half_width_s,
        upper=grid.max_delay_s,
    )
    rate_overlap = _cyclic_overlap(
        offset=rate_center - field.rate_center_hz,
        half_width=rate_half
        + max(footprint.rate_margin_hz, _rate_transition_hz(footprint, grid)),
        field_half_width=field.rate_half_width_hz,
        upper=grid.max_rate_hz,
    )
    return footprint.active & delay_overlap & rate_overlap


def _footprints_field_overlap(
    footprints: list[RateFootprint], field: RateBox, grid: _SegmentGrid
) -> NDArray[np.bool_]:
    """The times at which any of the footprints meets the field"""
    overlap = np.zeros(len(grid.time_s), dtype=bool)
    for footprint in footprints:
        overlap |= _footprint_field_overlap(footprint, field, grid)
    return overlap


def _footprint_notch(
    footprint: RateFootprint, grid: _SegmentGrid
) -> NDArray[np.floating[Any]]:
    """The taper nulling a footprint: zero across the region the object occupies at
    the times it is nulled, rising with a ``1 - cos`` transition beyond. Each time
    contributes a box in delay and fringe-rate, and the taper is the least of these.
    The occupied region is always zero, as the transition starts beyond it.

    Returns:
        NDArray[np.floating[Any]]: The taper, shape (rate, delay)
    """
    times = np.flatnonzero(footprint.active)
    nulled = np.zeros((len(grid.rate_hz), len(grid.delay_s)))
    if times.size == 0:
        return 1.0 - nulled

    rate_center, rate_half = footprint_rate_band(footprint, grid.max_rate_hz)
    rate_transition_hz = _rate_transition_hz(footprint, grid)
    # get_2d_taper returns one column per time, shape (axis, time)
    delay_null = 1.0 - get_2d_taper(
        x=grid.delay_s,
        outer_width=footprint.delay_occupied_half_s[times]
        + max(footprint.delay_margin_s, grid.delay_transition_s),
        tukey_width=np.full(times.size, grid.delay_transition_s),
        tukey_offset=footprint.delay_s[times],
        upper_limit=grid.max_delay_s,
    )
    rate_null = 1.0 - get_2d_taper(
        x=grid.rate_hz,
        outer_width=rate_half[times]
        + max(footprint.rate_margin_hz, rate_transition_hz),
        tukey_width=np.full(times.size, rate_transition_hz),
        tukey_offset=rate_center[times],
        upper_limit=grid.max_rate_hz,
    )

    # The amount nulled is the most any time nulls. Each time only touches a small
    # block of the plane, so only that block is updated
    for column in range(times.size):
        rate_rows = np.flatnonzero(rate_null[:, column])
        delay_cols = np.flatnonzero(delay_null[:, column])
        if rate_rows.size == 0 or delay_cols.size == 0:
            continue
        block = np.ix_(rate_rows, delay_cols)
        nulled[block] = np.maximum(
            nulled[block],
            rate_null[rate_rows, column][:, None]
            * delay_null[delay_cols, column][None, :],
        )
    return 1.0 - nulled


def _search_delay_shift(
    amplitude: NDArray[np.floating[Any]],
    footprints: list[RateFootprint],
    grid: _SegmentGrid,
    protection: NDArray[np.floating[Any]],
    search_half_width_s: float,
) -> float:
    """The shift in delay, within ``search_half_width_s``, that places an object's
    footprints on the most power away from the field, as the delay taper's peak
    search does. The object is separated from the field in fringe-rate here, so the
    field does not draw the search. Times nulled across all fringe-rates are not
    used, as they include the field's fringe-rates. Ties keep the smallest shift.

    Returns:
        float: The shift in delay, in seconds
    """
    nulled = np.zeros_like(amplitude)
    for footprint in footprints:
        in_rate = replace(footprint, active=footprint.active & ~footprint.all_rates)
        nulled = np.maximum(nulled, 1.0 - _footprint_notch(in_rate, grid))
    if not np.any(nulled):
        return 0.0

    weighted = amplitude * (1.0 - protection)
    delay_bin_s = float(np.abs(np.mean(np.diff(grid.delay_s))))
    max_shift = int(np.floor(search_half_width_s / delay_bin_s + 1e-9))
    best_shift, best_score = 0, float(np.sum(nulled * weighted))
    # Smallest shifts first, so that ties keep the smaller shift
    for shift in sorted(range(-max_shift, max_shift + 1), key=abs):
        score = float(np.sum(np.roll(nulled, shift, axis=1) * weighted))
        if score > best_score:
            best_shift, best_score = shift, score
    return best_shift * delay_bin_s


def _fit_object_footprints(
    w_delays: WDelays,
    segment: ContaminatedSegment,
    grid: _SegmentGrid,
    field: RateBox,
    settings: RateFilterSettings,
    candidate_widths_hz: list[float],
    amplitude: NDArray[np.floating[Any]],
    protection: NDArray[np.floating[Any]],
) -> tuple[list[RateFootprint], NDArray[np.bool_]]:
    """The footprints nulling an object across a segment: its main lobe and delay
    sidelobes, with the widest rate margin that keeps them separable from the field,
    shifted in delay should a peak search be requested.

    Returns:
        tuple[list[RateFootprint], NDArray[np.bool_]]: The footprints, empty if the object is not nulled at any time, and the times at which the object meets the field. Should it meet the field at any time no footprints are returned.
    """
    b_idx = segment.baseline_idx
    t_idx = np.array(segment.time_idx)
    tau_s = w_delays.w_delays[b_idx, t_idx].to(u.s).value
    tau_rate = get_w_rates(w_delays)[b_idx, t_idx].value

    # Nulled as the delay taper nulls: above the elevation cut, and not wrapped too far in delay
    active = (w_delays.elevation[t_idx] >= settings.elevation_cut_deg * u.deg) & (
        calculate_nyquist_zone(values=tau_s, upper_limit=grid.max_delay_s)
        <= settings.ignore_nyquist_zone
    )
    no_overlap = np.zeros(len(t_idx), dtype=bool)
    if not np.any(active):
        return [], no_overlap
    # Beyond the fringe-rate Nyquist zone the object can only be nulled in delay
    all_rates = active & _beyond_rate_nyquist_zone(
        tau_rate, grid.nu_min_hz, grid.max_rate_hz, settings.ignore_rate_nyquist_zone
    )

    footprints: list[RateFootprint] | None = None
    overlap = no_overlap
    for width_hz in candidate_widths_hz:
        candidate = _make_footprints(
            object_name=w_delays.object_name,
            tau_s=tau_s,
            tau_rate=tau_rate,
            active=active,
            all_rates=all_rates,
            grid=grid,
            settings=settings,
            rate_margin_hz=width_hz,
        )
        # The narrowest margin is tried last, so its overlap is the one reported
        overlap = _footprints_field_overlap(candidate, field, grid)
        if not np.any(overlap):
            footprints = candidate
            break
    if footprints is None:
        return [], overlap
    if footprints[0].rate_margin_hz < candidate_widths_hz[0]:
        logger.debug(
            f"Reduced the delay-rate margin of {w_delays.object_name} to {footprints[0].rate_margin_hz:.3g} Hz "
            f"(from {candidate_widths_hz[0]:.3g} Hz) to stay separable from the field"
        )

    if settings.peak_shift_search:
        search_width_ns = (
            settings.peak_shift_search_width_ns
            if settings.peak_shift_search_width_ns is not None
            else settings.outer_width_ns
        )
        shift_s = _search_delay_shift(
            amplitude=amplitude,
            footprints=footprints,
            grid=grid,
            protection=protection,
            search_half_width_s=search_width_ns * 1e-9,
        )
        if shift_s != 0.0:
            logger.debug(f"Shifted {w_delays.object_name} by {shift_s * 1e9:.2f} ns")
            footprints = [
                replace(footprint, delay_s=footprint.delay_s + shift_s)
                for footprint in footprints
            ]
            overlap = _footprints_field_overlap(footprints, field, grid)
            if np.any(overlap):
                return [], overlap

    return footprints, no_overlap


def rate_filter_segment(
    segment: ContaminatedSegment,
    freq_chan: u.Quantity,
    w_delays_list: list[WDelays],
    settings: RateFilterSettings,
    keep_diagnostics: bool = False,
) -> RateFilterResult:
    """Null the objects in ``w_delays_list`` from a segment in delay and
    delay-rate space while protecting the field.

    Each object is nulled over its footprint: at each time it is at its predicted
    delay and spans the fringe-rates across the band, along with its delay sidelobes
    should they be nulled. The field occupies a box around (delay, rate) = (0, 0)
    whose size is set by the guard regions. Should any footprint meet the field the
    object can not be separated, and the segment is not filtered. The timesteps that
    meet the field are reported, so ``rate_filter_segments`` can filter the rest.

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
        return RateFilterResult(rows=core_rows, outcome=RateFilterOutcome.TOO_SHORT)

    time_s = np.array(segment.time_mjds)
    data = np.array(segment.data)
    # Unflagged non-finite values would otherwise spread across the whole plane
    mask = np.array(segment.mask) | ~np.isfinite(data)
    delay_rate = array_to_delay_rate(
        masked_data=np.ma.masked_array(data, mask=mask),
        freq_chan=freq_chan,
        time_s=time_s,
    )
    delay_s = delay_rate.delay.to(u.s).value
    rate_hz = delay_rate.rate.to(u.Hz).value
    rate_bin_hz = 1.0 / (len(time_s) * float(np.mean(np.diff(time_s))))

    freq_hz = freq_chan.to(u.Hz).value
    outer_width_s = settings.outer_width_ns * 1e-9
    # Delay and fringe-rate are cyclic, wrapping at half of their period
    grid = _SegmentGrid(
        delay_s=delay_s,
        rate_hz=rate_hz,
        time_s=time_s,
        max_delay_s=axis_half_period(delay_s),
        max_rate_hz=axis_half_period(rate_hz),
        nu_min_hz=float(np.min(freq_hz)),
        nu_max_hz=float(np.max(freq_hz)),
        delay_transition_s=_transition_width(
            x=delay_s, requested=settings.tukey_width_ns * 1e-9, fallback=outer_width_s
        ),
    )
    candidate_widths_hz = _candidate_rate_widths(
        settings=settings,
        sinc_width_hz=calculate_expected_rate_sinc_width(time_s).to(u.Hz).value,
        rate_bin_hz=rate_bin_hz,
    )

    # The guard regions are common to all objects
    reference = w_delays_list[0]
    b_idx = segment.baseline_idx
    t_idx = np.array(segment.time_idx)
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
        rate_half_width_hz=grid.nu_max_hz * rate_guard
        + _rate_floor_hz(settings, rate_bin_hz),
    )
    # Never modify the field. The protection is one across the field and rolls
    # off smoothly, with the narrowest margin that may be used for an object, so
    # the taper has no discontinuity at the field's boundary
    field_rate_transition_hz = _transition_width(
        x=rate_hz,
        requested=candidate_widths_hz[-1],
        fallback=candidate_widths_hz[-1],
    )
    protection = (
        _axis_protection(
            x=rate_hz,
            half_width=field.rate_half_width_hz,
            tukey_width=field_rate_transition_hz,
        )[:, None]
        * _axis_protection(
            x=delay_s,
            half_width=field.delay_half_width_s,
            tukey_width=grid.delay_transition_s,
        )[None, :]
    )

    amplitude = np.abs(delay_rate.delay_rate).sum(axis=-1)
    footprints: list[RateFootprint] = []
    contaminated = np.zeros(segment.n_rows, dtype=bool)
    for w_delays in w_delays_list:
        object_footprints, overlap = _fit_object_footprints(
            w_delays=w_delays,
            segment=segment,
            grid=grid,
            field=field,
            settings=settings,
            candidate_widths_hz=candidate_widths_hz,
            amplitude=amplitude,
            protection=protection,
        )
        contaminated |= overlap
        footprints.extend(object_footprints)
    if np.any(contaminated):
        return RateFilterResult(
            rows=core_rows,
            outcome=RateFilterOutcome.RATE_CONTAMINATED,
            contaminated_timesteps=contaminated,
        )

    notch = np.ones((len(rate_hz), len(delay_s)))
    for footprint in footprints:
        notch = np.minimum(notch, _footprint_notch(footprint, grid))
    if np.all(notch == 1.0):
        return RateFilterResult(
            rows=core_rows, outcome=RateFilterOutcome.NOTHING_TO_NULL
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
            footprints=footprints,
            taper=notch,
        )
    filtered = delay_rate_to_array(delay_rate)[core]

    flags = mask[core] | ~np.isfinite(filtered)

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
        outcome=RateFilterOutcome.FILTERED,
        data=filtered,
        flags=flags,
        weights=scaled_weights,
        nulled_fraction=1.0 - mean_notch,
        diagnostics=diagnostics,
        notches=footprints,
    )


def _segment_row(segment: ContaminatedSegment, index: int) -> _Row:
    """A row held by a segment"""
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
    segment: ContaminatedSegment, exclude: NDArray[np.bool_]
) -> list[ContaminatedSegment]:
    """The runs of consecutive timesteps of a segment between those excluded, that
    hold at least one core row. Pad rows within a run are kept as padding."""
    kept = np.flatnonzero(~exclude)
    runs = np.split(kept, np.flatnonzero(np.diff(kept) > 1) + 1) if kept.size else []
    pieces: list[ContaminatedSegment] = []
    for run in runs:
        if all(segment.is_pad[i] for i in run):
            continue
        piece = ContaminatedSegment(
            ant_1=segment.ant_1, ant_2=segment.ant_2, baseline_idx=segment.baseline_idx
        )
        for i in run:
            _append_segment_row(piece, _segment_row(segment, i), segment.is_pad[i])
        pieces.append(piece)
    return pieces


def rate_filter_segments(
    segment: ContaminatedSegment,
    freq_chan: u.Quantity,
    w_delays_list: list[WDelays],
    settings: RateFilterSettings,
    keep_diagnostics: bool = False,
) -> list[RateFilterResult]:
    """Filter a segment in delay-rate. Timesteps where any object meets the field
    remain flagged, and the rest of the segment is split around them and filtered
    again. Each split removes at least one timestep, so this terminates.

    Args:
        segment (ContaminatedSegment): The rows to filter
        freq_chan (u.Quantity): The frequency of each channel
        w_delays_list (list[WDelays]): The objects to null
        settings (RateFilterSettings): Parameterisation of the filter
        keep_diagnostics (bool, optional): Attach the quantities needed to plot the filtering. Defaults to False.

    Returns:
        list[RateFilterResult]: The outcome of each part of the segment. Every core row is in exactly one.
    """
    results: list[RateFilterResult] = []
    pending = [segment]
    while pending:
        current = pending.pop(0)
        result = rate_filter_segment(
            segment=current,
            freq_chan=freq_chan,
            w_delays_list=w_delays_list,
            settings=settings,
            keep_diagnostics=keep_diagnostics,
        )
        excluded = result.contaminated_timesteps
        pieces = [] if excluded is None else _split_segment(current, excluded)
        if excluded is None or not pieces:
            results.append(result)
            continue

        # The timesteps meeting the field remain flagged
        excluded_core = excluded & current.core
        if np.any(excluded_core):
            results.append(
                RateFilterResult(
                    rows=np.array(current.rows)[excluded_core],
                    outcome=RateFilterOutcome.RATE_CONTAMINATED,
                )
            )
        pending.extend(pieces)
    return results


@dataclass
class RateFilterSummary:
    """Tally of the delay-rate filtering outcomes"""

    segments_filtered: int = 0
    """The number of segments that were filtered"""
    rows_filtered: int = 0
    """The number of core rows that were filtered"""
    not_filtered: Counter[str] = field(default_factory=Counter)
    """The number of segments not filtered, by outcome"""
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
    if result.outcome is RateFilterOutcome.FILTERED:
        summary.segments_filtered += 1
        summary.rows_filtered += len(result.rows)
    else:
        summary.not_filtered[result.outcome.value] += 1
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
    if summary.not_filtered:
        logger.info(
            f"Segments not filtered ({summary.rows_not_filtered} rows): {dict(summary.not_filtered)}"
        )
    logger.info(
        f"{doubly_contaminated_rows} rows were contaminated in delay and delay-rate"
    )
