from __future__ import annotations

from typing import Any

import astropy.units as u
import numpy as np
import pytest
from numpy.typing import NDArray

from jolly_roger.rates import (
    RateFilterResult,
    RateFilterSettings,
    RateFilterSummary,
    SegmentAccumulator,
    rate_filter_segment,
)
from jolly_roger.uvws import WDelays

N_CHAN = 64
DT_S = 10.0
T0_S = 5e9
FREQ_HZ = np.linspace(0.8e9, 1.1e9, N_CHAN)


def _update(
    accumulator: SegmentAccumulator,
    recoverable: NDArray[np.bool_],
    delay_contaminated: NDArray[np.bool_] | None = None,
    time_idx: NDArray[np.int_] | None = None,
    ant_2: NDArray[np.int_] | None = None,
    row_start: int = 0,
    data: NDArray[np.complexfloating[Any]] | None = None,
    weights: dict[str, NDArray[np.floating[Any]]] | None = None,
):
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

    return accumulator.update(
        row_numbers=row_start + np.arange(n_rows),
        ant_1=np.zeros(n_rows, dtype=int),
        ant_2=ant_2,
        baseline_idx=ant_2 - 1,
        time_mjds=T0_S + time_idx * DT_S,
        time_idx=time_idx,
        data=data,
        mask=np.zeros(data.shape, dtype=bool),
        recoverable=np.asarray(recoverable, dtype=bool),
        delay_contaminated=np.asarray(delay_contaminated, dtype=bool),
        weights=weights,
    )


def _bools(pattern: str) -> NDArray[np.bool_]:
    """Turn a string of '0'/'1' into a boolean array"""
    return np.array([c == "1" for c in pattern])


def test_accumulator_releases_on_exit() -> None:
    accumulator = SegmentAccumulator()
    released = _update(accumulator, _bools("0011100"))

    assert len(released) == 1
    assert released[0].rows == [2, 3, 4]
    assert released[0].n_core == 3
    assert accumulator.flush_all() == []


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
    released += accumulator.flush_all()

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
    released += accumulator.flush_all()

    assert [seg.rows for seg in released] == [[0, 1], [2, 3]]


def test_accumulator_releases_on_max_timesteps() -> None:
    accumulator = SegmentAccumulator(max_timesteps=3)
    released = _update(accumulator, _bools("1111111"))
    released += accumulator.flush_all()

    assert [seg.rows for seg in released] == [[0, 1, 2], [3, 4, 5], [6]]


def test_accumulator_flush_all() -> None:
    accumulator = SegmentAccumulator()
    assert _update(accumulator, _bools("0011")) == []

    released = accumulator.flush_all()
    assert [seg.rows for seg in released] == [[2, 3]]
    assert accumulator.flush_all() == []


def test_accumulator_copies_rows() -> None:
    """Rows held across chunks should not reference the chunk's arrays"""
    accumulator = SegmentAccumulator()
    data = np.ones((2, 4, 2), dtype=complex)
    weights = {"WEIGHT": np.ones((2, 2))}
    _update(accumulator, _bools("11"), data=data, weights=weights)
    data[:] = 0
    weights["WEIGHT"][:] = 0

    (segment,) = accumulator.flush_all()
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
    released += accumulator.flush_all()

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
    released += accumulator.flush_all()

    assert [seg.rows for seg in released] == [[2, 3, 4]]


def test_accumulator_partial_trail_flushed() -> None:
    accumulator = SegmentAccumulator(pad_timesteps=3)
    assert _update(accumulator, _bools("0110")) == []

    (segment,) = accumulator.flush_all()
    assert segment.rows == [0, 1, 2, 3]
    assert segment.core_rows.tolist() == [1, 2]


def _make_segment_inputs(
    n_time: int = 64,
    tau_rate: float = 2e-11,
    object_amp: float = 5.0,
    sign: float = 1.0,
) -> tuple[NDArray[np.complexfloating[Any]], NDArray[np.complexfloating[Any]], WDelays]:
    """A field source at (delay, rate)=(0, 0) and an object whose delay
    crosses zero at a constant delay-rate"""
    time_s = T0_S + np.arange(n_time) * DT_S
    tau_s = tau_rate * (time_s - time_s.mean())

    object_vis = object_amp * np.exp(
        sign * 2j * np.pi * FREQ_HZ[None, :] * tau_s[:, None]
    )
    data = np.repeat((1.0 + object_vis)[..., None], 2, axis=-1)

    w_delays = WDelays(
        object_name="sun",
        w_delays=(tau_s * u.s)[None, :],
        b_map={(0, 1): 0},
        time_map={t * u.s: idx for idx, t in enumerate(time_s)},
        elevation=np.full(n_time, 45.0) * u.deg,
    )

    return data, np.repeat(object_vis[..., None], 2, axis=-1), w_delays


def _filter(
    data: NDArray[np.complexfloating[Any]],
    w_delays: WDelays,
    recoverable: NDArray[np.bool_],
    pad_timesteps: int = 0,
    settings: RateFilterSettings | None = None,
    weights: dict[str, NDArray[np.floating[Any]]] | None = None,
) -> tuple[RateFilterResult, NDArray[np.bool_]]:
    accumulator = SegmentAccumulator(pad_timesteps=pad_timesteps)
    released = _update(accumulator, recoverable, data=data, weights=weights)
    released += accumulator.flush_all()
    assert len(released) == 1

    result = rate_filter_segment(
        segment=released[0],
        freq_chan=FREQ_HZ * u.Hz,
        w_delays_list=[w_delays],
        settings=settings
        or RateFilterSettings(outer_width_ns=10.0, tukey_width_ns=5.0),
    )
    return result, released[0].core


def _suppression_db(
    result: RateFilterResult, object_vis: NDArray[np.complexfloating[Any]]
) -> float:
    assert result.data is not None
    residual = result.data - 1.0
    original = object_vis[result.rows]
    return float(
        10 * np.log10(np.sum(np.abs(residual) ** 2) / np.sum(np.abs(original) ** 2))
    )


def test_rate_filter_nulls_object() -> None:
    """The object should be removed while the field is kept. This also sets
    the sign convention between the delay-rate transform and ``w_rates``."""
    data, object_vis, w_delays = _make_segment_inputs()
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))

    assert result.success
    assert result.data is not None
    assert result.data.shape == data.shape
    assert _suppression_db(result, object_vis) < -10.0
    assert np.abs(np.mean(result.data) - 1.0) < 0.05
    assert 0.0 < result.nulled_fraction < 0.1


def test_rate_filter_wrong_sign_does_not_null() -> None:
    """Guard against the sign convention silently flipping"""
    data, object_vis, w_delays = _make_segment_inputs(sign=-1.0)
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))

    assert result.success
    assert _suppression_db(result, object_vis) > -1.0


def test_rate_filter_preserves_field() -> None:
    """Data only containing the field are not modified"""
    data, _, w_delays = _make_segment_inputs(object_amp=0.0)
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))

    assert result.success
    assert result.data is not None
    np.testing.assert_allclose(result.data, data, atol=1e-10)


def test_rate_filter_scales_weights() -> None:
    data, _, w_delays = _make_segment_inputs()
    weights = {"WEIGHT": np.ones((len(data), 2))}
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool), weights=weights)

    assert result.weights is not None
    expected = 1.0 / (1.0 - result.nulled_fraction)
    np.testing.assert_allclose(result.weights["WEIGHT"], expected)


def test_rate_filter_rate_contaminated() -> None:
    """An object with no delay-rate can not be separated from the field"""
    data, _, w_delays = _make_segment_inputs(tau_rate=1e-14)
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))

    assert not result.success
    assert result.reason == "rate-contaminated"
    assert result.data is None


def test_rate_filter_rate_contaminated_by_guard() -> None:
    """The geometric guard region in rate protects the field"""
    data, _, w_delays = _make_segment_inputs()
    n_time = len(data)
    # A field fringe-rate of ~0.03 Hz at 1.1 GHz exceeds the object's
    guarded = WDelays(
        object_name=w_delays.object_name,
        w_delays=w_delays.w_delays,
        b_map=w_delays.b_map,
        time_map=w_delays.time_map,
        elevation=w_delays.elevation,
        rate_guard_region=np.full((1, n_time), 3e-11) * u.dimensionless_unscaled,
    )
    result, _ = _filter(data, guarded, np.ones(n_time, dtype=bool))

    assert not result.success
    assert result.reason == "rate-contaminated"


def test_rate_filter_too_short() -> None:
    data, _, w_delays = _make_segment_inputs()
    recoverable = np.zeros(len(data), dtype=bool)
    recoverable[30:34] = True
    result, _ = _filter(data, w_delays, recoverable)

    assert not result.success
    assert result.reason == "too short"
    assert result.rows.tolist() == [30, 31, 32, 33]


def test_rate_filter_below_elevation() -> None:
    data, _, w_delays = _make_segment_inputs()
    below = WDelays(
        object_name=w_delays.object_name,
        w_delays=w_delays.w_delays,
        b_map=w_delays.b_map,
        time_map=w_delays.time_map,
        elevation=np.full(len(data), -10.0) * u.deg,
    )
    result, _ = _filter(data, below, np.ones(len(data), dtype=bool))

    assert not result.success
    assert result.reason == "nothing to null"


@pytest.mark.parametrize("pad_timesteps", [0, 16])
def test_rate_filter_padding_resolves_short_segment(pad_timesteps: int) -> None:
    """Too few timesteps can not resolve the object from the field in rate,
    but padding with clean neighbours can"""
    data, object_vis, w_delays = _make_segment_inputs()
    recoverable = np.zeros(len(data), dtype=bool)
    recoverable[28:36] = True
    result, _ = _filter(data, w_delays, recoverable, pad_timesteps=pad_timesteps)

    # Only the core rows are returned to be written
    assert result.rows.tolist() == list(range(28, 36))
    if pad_timesteps == 0:
        assert not result.success
        assert result.reason == "rate-contaminated"
    else:
        assert result.success
        assert _suppression_db(result, object_vis) < -15.0


def test_rate_filter_summary() -> None:
    summary = RateFilterSummary()
    summary.record(RateFilterResult(rows=np.arange(4), success=True))
    summary.record(
        RateFilterResult(rows=np.arange(2), success=False, reason="too short")
    )

    assert summary.segments_filtered == 1
    assert summary.rows_filtered == 4
    assert summary.failures["too short"] == 1
    assert summary.rows_not_filtered == 2
    summary.log(doubly_contaminated_rows=3)
