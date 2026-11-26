"""Collect the rows of each baseline that delay-rate filtering can recover.

The delay notch flags a row when an object's delay is too close to the
field's to separate the two in delay. Many of those rows can still be
separated in fringe rate: over a run of consecutive timesteps the object and
the field move differently in delay, so a two-dimensional (delay and
fringe-rate) filter can remove the object while keeping the field.

A measurement set is read in time-ordered chunks of rows from many
baselines, and the timesteps of one baseline that need such a filter may span
several chunks. ``update_segment_accumulator`` therefore keeps, for each
baseline, the run of consecutive rows that are

- flagged by the delay notch (``delay_contaminated``), and
- recoverable: every object near the field in delay is separated from the
  field in fringe rate,

and releases the run as a ``ContaminatedSegment`` once it ends: when the
baseline next sees a row that is not flagged, a row that is not recoverable,
a gap in time, or ``max_timesteps`` rows. ``flush_segment_accumulator``
releases whatever remains at the end of the measurement set.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

GAP_FACTOR = 1.5
"""A step in time of more than this many integrations ends a segment"""


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
    """The consecutive timesteps of a baseline to filter in delay and fringe
    rate. Core rows are those flagged by the delay notch. Pad rows are clean
    neighbouring timesteps that only improve the fringe-rate resolution and
    are never written back."""

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
    """Whether a row is padding rather than a core row"""
    n_core: int = 0
    """The number of core rows"""
    weights: dict[str, list[NDArray[np.floating[Any]]]] | None = None
    """The original weights of each row, keyed by column name"""

    @property
    def n_rows(self) -> int:
        """The total number of rows, core and pad"""
        return len(self.rows)

    @property
    def core(self) -> NDArray[np.bool_]:
        """Mask selecting the core rows"""
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
    time_step_s: float | None = None
    """The smallest step in time seen between consecutive rows, in seconds"""


@dataclass
class SegmentAccumulator:
    """Collect, for each baseline, the consecutive rows that delay-rate
    filtering can recover, and release them as a segment once the run ends.
    Rows are consumed by ``update_segment_accumulator``.

    Each row is one of:
    - clean: not flagged by the delay notch
    - recoverable: flagged by the delay notch, and every object near the field in delay is separated from it in fringe rate
    - not recoverable: flagged by the delay notch, and some object is near the field in both delay and fringe rate

    A segment is released when its baseline sees a clean row (after
    ``pad_timesteps`` trailing clean rows have been collected), a row that is
    not recoverable, a gap in time, or ``max_timesteps`` core rows. Up to
    ``pad_timesteps`` clean rows before a segment are added in front of it.
    """

    max_timesteps: int | None = 256
    """The most core rows of a segment before it is released. Bounds memory: each baseline being collected holds up to (``max_timesteps`` + 2 ``pad_timesteps``) rows of channels x polarisations x 16 bytes. If None there is no limit."""
    pad_timesteps: int = 0
    """The number of clean rows either side of a segment to include as padding"""
    max_time_gap_s: float | None = None
    """The largest step in time between consecutive rows of a baseline within a segment, in seconds, e.g. ``GAP_FACTOR`` times the integration time. If None it is ``GAP_FACTOR`` times the smallest step seen so far on that baseline, which misses a gap between the first two rows. A row that skips a time index also ends a segment."""
    states: dict[tuple[int, int], _BaselineState] = field(default_factory=dict)
    """The tracking of each baseline, keyed by (ANTENNA1, ANTENNA2)"""
    doubly_contaminated_rows: int = 0
    """Count of rows flagged by the delay notch that could not be recovered because an object is near the field in fringe rate as well"""


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
    """Rows flagged by the delay notch because an object is too close to the field in delay"""
    recoverable: NDArray[np.bool_]
    """Rows flagged by the delay notch where every object near the field in delay is separated from it in fringe rate"""
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
    """Whether a row does not directly follow the previous row of its
    baseline: it skips a time index, or the step in time is larger than
    ``max_time_gap_s`` (or, if None, ``GAP_FACTOR`` times the smallest step
    seen so far on the baseline). Updates the smallest step seen."""
    if state.last_time_idx is None or state.last_time_mjd is None:
        return False
    step_s = time_mjd - state.last_time_mjd
    if not 0 < time_idx - state.last_time_idx <= 1 or step_s <= 0:
        return True
    if state.time_step_s is None or step_s < state.time_step_s:
        state.time_step_s = step_s
    limit_s = (
        GAP_FACTOR * state.time_step_s if max_time_gap_s is None else max_time_gap_s
    )
    return step_s > limit_s


def update_segment_accumulator(
    accumulator: SegmentAccumulator, payload: RateFilterPayload
) -> list[ContaminatedSegment]:
    """Consume the rows of a chunk, in the order they appear in the measurement set.

    Args:
        accumulator (SegmentAccumulator): The per-baseline collection of rows, updated in place
        payload (RateFilterPayload): The rows, and whether each is flagged by the delay notch and recoverable

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
