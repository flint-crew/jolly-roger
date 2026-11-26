"""Delay-rate filtering of rows the delay notch flags"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


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
    rows), a flagged row that is not a candidate, a gap in time, or
    ``max_timesteps`` candidates."""

    max_timesteps: int | None = 256
    """The most candidates in a segment. Bounds memory. If None there is no limit."""
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
    original_weights: dict[str, NDArray[np.floating[Any]]] | None = None
    """The weights before any scaling"""


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

            if (
                max_timesteps is not None
                and state.segment.n_candidates >= max_timesteps
            ):
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
