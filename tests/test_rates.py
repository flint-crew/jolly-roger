from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from jolly_roger.rates import (
    ContaminatedSegment,
    RateFilterPayload,
    SegmentAccumulator,
    flush_segment_accumulator,
    update_segment_accumulator,
)

DT_S = 10.0
T0_S = 5e9


def _update(
    accumulator: SegmentAccumulator,
    recoverable: NDArray[np.bool_],
    delay_contaminated: NDArray[np.bool_] | None = None,
    time_idx: NDArray[np.int_] | None = None,
    ant_2: NDArray[np.int_] | None = None,
    row_start: int = 0,
    data: NDArray[np.complexfloating[Any]] | None = None,
    weights: dict[str, NDArray[np.floating[Any]]] | None = None,
    time_s: NDArray[np.floating[Any]] | None = None,
) -> list[ContaminatedSegment]:
    """Feed a set of rows to the accumulator. By default rows are a single
    baseline at consecutive times"""
    n_rows = len(recoverable)
    if delay_contaminated is None:
        delay_contaminated = recoverable
    if time_idx is None:
        time_idx = row_start + np.arange(n_rows)
    if ant_2 is None:
        ant_2 = np.ones(n_rows, dtype=int)
    if data is None:
        data = np.ones((n_rows, 4, 2), dtype=complex)
    if time_s is None:
        time_s = T0_S + time_idx * DT_S

    payload = RateFilterPayload(
        rows=row_start + np.arange(n_rows),
        ant_1=np.zeros(n_rows, dtype=int),
        ant_2=ant_2,
        time_mjds=time_s,
        baseline_idx=ant_2 - 1,
        time_idx=time_idx,
        original_masked_data=np.ma.masked_array(data, mask=np.zeros(data.shape, bool)),
        delay_contaminated=np.asarray(delay_contaminated, dtype=bool),
        recoverable=np.asarray(recoverable, dtype=bool),
        original_weights=weights,
    )
    return update_segment_accumulator(accumulator=accumulator, payload=payload)


def _bools(pattern: str) -> NDArray[np.bool_]:
    """Turn a string of '0'/'1' into a boolean array"""
    return np.array([c == "1" for c in pattern])


def test_accumulator_releases_on_exit() -> None:
    accumulator = SegmentAccumulator()
    released = _update(accumulator, _bools("0011100"))

    assert len(released) == 1
    assert released[0].rows == [2, 3, 4]
    assert released[0].n_core == 3
    assert flush_segment_accumulator(accumulator) == []


def test_accumulator_releases_across_chunks() -> None:
    """A baseline that enters the zone in one chunk and exits in another"""
    accumulator = SegmentAccumulator()
    assert _update(accumulator, _bools("0011"), row_start=0) == []
    released = _update(accumulator, _bools("1100"), row_start=4)

    assert len(released) == 1
    assert released[0].rows == [2, 3, 4, 5]


def test_accumulator_tracks_baselines_independently() -> None:
    """Rows interleaved from two baselines, each time step holds both"""
    accumulator = SegmentAccumulator()
    # (baseline 1, baseline 2) per time step
    recoverable = _bools("10111101")
    ant_2 = np.tile([1, 2], 4)
    time_idx = np.repeat(np.arange(4), 2)
    released = _update(accumulator, recoverable, time_idx=time_idx, ant_2=ant_2)
    released += flush_segment_accumulator(accumulator)

    by_baseline = {(seg.ant_1, seg.ant_2): seg.rows for seg in released}
    assert by_baseline[(0, 1)] == [0, 2, 4]
    assert by_baseline[(0, 2)] == [3, 5, 7]


def test_accumulator_releases_on_doubly_contaminated() -> None:
    accumulator = SegmentAccumulator()
    recoverable = _bools("0110110")
    delay_contaminated = _bools("0111110")
    released = _update(accumulator, recoverable, delay_contaminated)

    assert [seg.rows for seg in released] == [[1, 2], [4, 5]]
    assert accumulator.doubly_contaminated_rows == 1


def test_accumulator_releases_on_time_gap() -> None:
    accumulator = SegmentAccumulator()
    released = _update(accumulator, _bools("1111"), time_idx=np.array([0, 1, 3, 4]))
    released += flush_segment_accumulator(accumulator)

    assert [seg.rows for seg in released] == [[0, 1], [2, 3]]


def test_accumulator_releases_on_max_timesteps() -> None:
    accumulator = SegmentAccumulator(max_timesteps=3)
    released = _update(accumulator, _bools("1111111"))
    released += flush_segment_accumulator(accumulator)

    assert [seg.rows for seg in released] == [[0, 1, 2], [3, 4, 5], [6]]


def test_accumulator_flush_all() -> None:
    accumulator = SegmentAccumulator()
    assert _update(accumulator, _bools("0011")) == []

    released = flush_segment_accumulator(accumulator)
    assert [seg.rows for seg in released] == [[2, 3]]
    assert flush_segment_accumulator(accumulator) == []


def test_accumulator_copies_rows() -> None:
    """Rows held across chunks should not reference the chunk's arrays"""
    accumulator = SegmentAccumulator()
    data = np.ones((2, 4, 2), dtype=complex)
    weights = {"WEIGHT": np.ones((2, 2))}
    _update(accumulator, _bools("11"), data=data, weights=weights)
    data[:] = 0
    weights["WEIGHT"][:] = 0

    (segment,) = flush_segment_accumulator(accumulator)
    assert np.all(np.array(segment.data) == 1)
    assert segment.weights is not None
    assert np.all(np.array(segment.weights["WEIGHT"]) == 1)


def test_accumulator_padding() -> None:
    accumulator = SegmentAccumulator(pad_timesteps=2)
    released = _update(accumulator, _bools("000111000"))

    assert len(released) == 1
    segment = released[0]
    assert segment.rows == [1, 2, 3, 4, 5, 6, 7]
    assert segment.is_pad == [True, True, False, False, False, True, True]
    assert segment.core_rows.tolist() == [3, 4, 5]


def test_accumulator_padding_shared_between_segments() -> None:
    """Clean rows between two segments trail the first and lead the second"""
    accumulator = SegmentAccumulator(pad_timesteps=2)
    released = _update(accumulator, _bools("110110"))
    released += flush_segment_accumulator(accumulator)

    assert [seg.rows for seg in released] == [[0, 1, 2], [2, 3, 4, 5]]
    assert [seg.core_rows.tolist() for seg in released] == [[0, 1], [3, 4]]


def test_accumulator_padding_cut_by_doubly_contaminated() -> None:
    accumulator = SegmentAccumulator(pad_timesteps=2)
    recoverable = _bools("0001100")
    delay_contaminated = _bools("0101100")
    released = _update(accumulator, recoverable, delay_contaminated)

    assert len(released) == 1
    # Row 1 is doubly contaminated so only row 2 may lead
    assert released[0].rows == [2, 3, 4, 5, 6]
    assert released[0].core_rows.tolist() == [3, 4]


def test_accumulator_padding_cut_by_time_gap() -> None:
    accumulator = SegmentAccumulator(pad_timesteps=2)
    released = _update(accumulator, _bools("00110"), time_idx=np.array([0, 1, 3, 4, 5]))
    released += flush_segment_accumulator(accumulator)

    assert [seg.rows for seg in released] == [[2, 3, 4]]


def test_accumulator_partial_trail_flushed() -> None:
    accumulator = SegmentAccumulator(pad_timesteps=3)
    assert _update(accumulator, _bools("0110")) == []

    (segment,) = flush_segment_accumulator(accumulator)
    assert segment.rows == [0, 1, 2, 3]
    assert segment.core_rows.tolist() == [1, 2]


def test_accumulator_releases_on_gap_in_seconds() -> None:
    """A step in time much longer than the integration ends a segment, even
    when the time index does not skip (e.g. between scans)"""
    accumulator = SegmentAccumulator()
    time_s = T0_S + DT_S * np.array([0, 1, 2, 30, 31, 32])
    released = _update(accumulator, _bools("111111"), time_s=time_s)
    released += flush_segment_accumulator(accumulator)

    assert [seg.rows for seg in released] == [[0, 1, 2], [3, 4, 5]]


def test_accumulator_releases_on_given_gap_in_seconds() -> None:
    """A given largest step in time is used from the first pair of rows"""
    accumulator = SegmentAccumulator(max_time_gap_s=1.5 * DT_S)
    time_s = T0_S + DT_S * np.array([0, 5, 6, 7])
    released = _update(accumulator, _bools("1111"), time_s=time_s)
    released += flush_segment_accumulator(accumulator)

    assert [seg.rows for seg in released] == [[0], [1, 2, 3]]


def test_accumulator_default_cap() -> None:
    """Segments are capped by default so memory stays bounded"""
    accumulator = SegmentAccumulator()
    assert accumulator.max_timesteps is not None
    released = _update(accumulator, np.ones(accumulator.max_timesteps + 1, bool))
    released += flush_segment_accumulator(accumulator)

    assert [seg.n_core for seg in released] == [accumulator.max_timesteps, 1]
