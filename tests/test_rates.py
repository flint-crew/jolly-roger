from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pytest
from numpy.typing import NDArray

from jolly_roger.deconvolve import rate_clean_segment
from jolly_roger.rates import (
    CandidateSegment,
    RateFilterPayload,
    RateFilterSettings,
    SegmentAccumulator,
    SegmentFilterResult,
    SegmentGeometry,
    SegmentOutcome,
    filter_segment,
    flush_rate_filter,
    flush_segment_accumulator,
    make_rate_filter_run,
    update_rate_filter,
    update_segment_accumulator,
)
from jolly_roger.tractor import (
    DataChunk,
    TukeyTractorOptions,
    compute_tukey_multi_taper,
)
from jolly_roger.uvws import WDelays

DT_S = 10.0
T0_S = 5e9


def _update(
    accumulator: SegmentAccumulator,
    candidate: NDArray[np.bool_],
    delay_contaminated: NDArray[np.bool_] | None = None,
    time_idx: NDArray[np.int_] | None = None,
    ant_2: NDArray[np.int_] | None = None,
    row_start: int = 0,
    data: NDArray[np.complexfloating[Any]] | None = None,
    weights: dict[str, NDArray[np.floating[Any]]] | None = None,
    time_s: NDArray[np.floating[Any]] | None = None,
    needed_s: float | None = None,
) -> list[CandidateSegment]:
    """Feed a set of rows to the accumulator. By default rows are a single
    baseline at consecutive times"""
    n_rows = len(candidate)
    if delay_contaminated is None:
        delay_contaminated = candidate
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
        candidate=np.asarray(candidate, dtype=bool),
        needed_s=None if needed_s is None else np.full(n_rows, needed_s),
        original_weights=weights,
    )
    return update_segment_accumulator(accumulator=accumulator, payload=payload)


def _candidate_rows(segment: CandidateSegment) -> list[int]:
    return [
        row for row, pad in zip(segment.rows, segment.is_pad, strict=True) if not pad
    ]


def _bools(pattern: str) -> NDArray[np.bool_]:
    """Turn a string of '0'/'1' into a boolean array"""
    return np.array([c == "1" for c in pattern])


def test_accumulator_releases_on_exit() -> None:
    accumulator = SegmentAccumulator()
    released = _update(accumulator, _bools("0011100"))

    assert len(released) == 1
    assert released[0].rows == [2, 3, 4]
    assert released[0].n_candidates == 3
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
    candidate = _bools("10111101")
    ant_2 = np.tile([1, 2], 4)
    time_idx = np.repeat(np.arange(4), 2)
    released = _update(accumulator, candidate, time_idx=time_idx, ant_2=ant_2)
    released += flush_segment_accumulator(accumulator)

    by_baseline = {(seg.ant_1, seg.ant_2): seg.rows for seg in released}
    assert by_baseline[(0, 1)] == [0, 2, 4]
    assert by_baseline[(0, 2)] == [3, 5, 7]


def test_accumulator_releases_on_doubly_contaminated() -> None:
    accumulator = SegmentAccumulator()
    candidate = _bools("0110110")
    delay_contaminated = _bools("0111110")
    released = _update(accumulator, candidate, delay_contaminated)

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
    assert _candidate_rows(segment) == [3, 4, 5]


def test_accumulator_padding_shared_between_segments() -> None:
    """Clean rows between two segments trail the first and lead the second"""
    accumulator = SegmentAccumulator(pad_timesteps=2)
    released = _update(accumulator, _bools("110110"))
    released += flush_segment_accumulator(accumulator)

    assert [seg.rows for seg in released] == [[0, 1, 2], [2, 3, 4, 5]]
    assert [_candidate_rows(seg) for seg in released] == [[0, 1], [3, 4]]


def test_accumulator_padding_cut_by_doubly_contaminated() -> None:
    accumulator = SegmentAccumulator(pad_timesteps=2)
    candidate = _bools("0001100")
    delay_contaminated = _bools("0101100")
    released = _update(accumulator, candidate, delay_contaminated)

    assert len(released) == 1
    # Row 1 is doubly contaminated so only row 2 may lead
    assert released[0].rows == [2, 3, 4, 5, 6]
    assert _candidate_rows(released[0]) == [3, 4]


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
    assert _candidate_rows(segment) == [1, 2]


def test_accumulator_releases_on_given_gap_in_seconds() -> None:
    """A step in time with no skipped index, e.g. between scans"""
    accumulator = SegmentAccumulator(max_time_gap_s=1.5 * DT_S)
    time_s = T0_S + DT_S * np.array([0, 5, 6, 7])
    released = _update(accumulator, _bools("1111"), time_s=time_s)
    released += flush_segment_accumulator(accumulator)

    assert [seg.rows for seg in released] == [[0], [1, 2, 3]]


def test_accumulator_releases_on_needed_duration() -> None:
    """A long run of candidates is split once a segment lasts twice the
    duration it needs, with segments overlapping so every candidate is
    written once and is away from the ends of its segment"""
    accumulator = SegmentAccumulator(pad_timesteps=2)
    pattern = "00" + "1" * 20 + "00"
    released = _update(accumulator, _bools(pattern), needed_s=3 * DT_S)
    released += flush_segment_accumulator(accumulator)

    # Released once it spans 60 s (2 x 30 s), handing its last 2 candidates
    # and the 2 rows before them to the next segment
    first, second = released[0], released[1]
    assert first.rows == [0, 1, 2, 3, 4, 5, 6]
    assert _candidate_rows(first) == [2, 3, 4]
    assert second.rows[:4] == [3, 4, 5, 6]
    assert _candidate_rows(second)[:2] == [5, 6]
    written = [row for segment in released for row in _candidate_rows(segment)]
    assert sorted(written) == list(range(2, 22))
    assert len(written) == len(set(written))
    for segment in released:
        candidates = _candidate_rows(segment)
        assert segment.rows[0] not in candidates
        assert segment.rows[-1] not in candidates


def test_accumulator_without_needed_duration() -> None:
    """Without a needed duration a run of candidates is one segment"""
    accumulator = SegmentAccumulator(pad_timesteps=2)
    released = _update(accumulator, _bools("0" + "1" * 50 + "0"))
    released += flush_segment_accumulator(accumulator)
    assert [seg.n_candidates for seg in released] == [50]


N_CHAN = 288
CHANNEL_HZ = 1e6
FREQ_HZ = 0.8e9 + CHANNEL_HZ * np.arange(N_CHAN)
OBJECT_RATE = 2e-11


def _segment(
    n_candidates: int,
    n_pad: int,
    object_rate: float = OBJECT_RATE,
    object_amplitude: float = 100.0,
    noise: float = 0.1,
) -> tuple[Any, ...]:
    """A baseline segment with a bright object crossing the field in delay,
    and a field of two static sources"""
    rng = np.random.default_rng(42)
    n_rows = n_candidates + 2 * n_pad
    time_s = T0_S + DT_S * np.arange(n_rows)
    delay_s = object_rate * (time_s - time_s[n_rows // 2]) + 0.5e-9
    phase = 2j * np.pi * FREQ_HZ[None, :]
    obj = object_amplitude * np.exp(phase * delay_s[:, None])
    field = np.exp(phase * 3e-9) + 0.5 * np.exp(phase * -20e-9)
    field = np.broadcast_to(field, obj.shape)
    shape = (*obj.shape, 1)
    data = (obj + field)[..., None] + noise * (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    )
    is_pad = np.ones(n_rows, dtype=bool)
    is_pad[n_pad : n_pad + n_candidates] = False
    geometry = SegmentGeometry(
        object_names=("sun",),
        object_delay_s=delay_s[None],
        object_rate=np.full((1, n_rows), object_rate),
        object_active=np.ones((1, n_rows), dtype=bool),
        field_delay_s=np.full(n_rows, 30e-9),
        field_rate=np.full(n_rows, 2e-13),
    )
    return data, is_pad, time_s, geometry, obj, field


def _filter(data, is_pad, time_s, geometry, flags=None) -> SegmentFilterResult:
    return filter_segment(
        data=data,
        flags=np.zeros(data.shape, bool) if flags is None else flags,
        is_pad=is_pad,
        time_s=time_s,
        freq_hz=FREQ_HZ,
        geometry=geometry,
        delay_width_s=10e-9,
        delay_roll_off_s=10e-9,
    )


def _delay_notch_residual(obj, geometry) -> NDArray[np.floating[Any]]:
    """RMS of the object left by the delay notch alone, per row"""
    delays_s = np.fft.fftfreq(N_CHAN, CHANNEL_HZ)
    half_period = 0.5 / CHANNEL_HZ
    offset = (delays_s[None, :] - geometry.object_delay_s[0][:, None] + half_period) % (
        2 * half_period
    ) - half_period
    distance = np.abs(offset) - 10e-9
    notch = 1 - 0.5 * (1 + np.cos(np.pi * np.clip(distance / 10e-9, 0, 1)))
    left = np.fft.ifft(np.fft.fft(obj, axis=1) * notch, axis=1)
    return np.sqrt(np.mean(np.abs(left) ** 2, axis=1))


def test_filter_segment_recovers_rows() -> None:
    """Recovered rows keep the field, and have no more of the object than the
    delay notch alone leaves"""
    data, is_pad, time_s, geometry, obj, field = _segment(n_candidates=60, n_pad=16)
    result = _filter(data, is_pad, time_s, geometry)

    assert result.outcome == SegmentOutcome.FILTERED
    assert result.recovered is not None
    assert result.data is not None
    assert not result.recovered[is_pad].any()
    assert result.recovered[~is_pad].mean() > 0.6

    left = np.sqrt(np.mean(np.abs(result.data[..., 0] - field) ** 2, axis=1))
    reference = _delay_notch_residual(obj, geometry)
    recovered = result.recovered
    assert np.median(left[recovered]) < np.median(reference[recovered])
    assert np.all(left[recovered] < 2 * reference[recovered] + 1)


def test_filter_segment_edge_rows_need_padding() -> None:
    """Without clean rows either side the candidates at the ends of the segment,
    where sidelobes of the finite segment are largest, are not recovered"""
    data, is_pad, time_s, geometry, *_ = _segment(n_candidates=60, n_pad=0)
    result = _filter(data, is_pad, time_s, geometry)
    assert result.recovered is not None
    assert not result.recovered[:16].any()
    assert not result.recovered[-16:].any()

    data, is_pad, time_s, geometry, *_ = _segment(n_candidates=60, n_pad=16)
    result = _filter(data, is_pad, time_s, geometry)
    assert result.recovered is not None
    candidates = np.flatnonzero(~is_pad)
    assert result.recovered[candidates[:8]].any()
    assert result.recovered[candidates[-8:]].any()


def test_filter_segment_keeps_field() -> None:
    """With no object present the field is unchanged in recovered rows"""
    data, is_pad, time_s, geometry, _, field = _segment(
        n_candidates=60, n_pad=16, object_amplitude=0.0, noise=0.0
    )
    result = _filter(data, is_pad, time_s, geometry)
    assert result.recovered is not None
    assert result.data is not None
    change = np.abs(result.data[..., 0] - field)[result.recovered]
    assert change.max() < 0.05


def test_filter_segment_not_separable() -> None:
    """An object with the field's fringe rate is left flagged"""
    data, is_pad, time_s, geometry, *_ = _segment(
        n_candidates=60, n_pad=16, object_rate=1e-13
    )
    assert (
        _filter(data, is_pad, time_s, geometry).outcome == SegmentOutcome.NOT_SEPARABLE
    )


def test_filter_segment_too_short() -> None:
    data, is_pad, time_s, geometry, *_ = _segment(n_candidates=1, n_pad=0)
    assert _filter(data, is_pad, time_s, geometry).outcome == SegmentOutcome.TOO_SHORT


def test_filter_segment_with_interior_flags() -> None:
    """Flagged values get zero weight, so do not reach unflagged channels"""
    data, is_pad, time_s, geometry, obj, field = _segment(n_candidates=60, n_pad=16)
    flags = np.zeros(data.shape, dtype=bool)
    flags[40, 100:140] = True
    data[flags] = 1e6
    result = _filter(data, is_pad, time_s, geometry, flags=flags)
    assert result.recovered is not None
    assert result.data is not None

    left = np.sqrt(
        np.mean(
            np.where(flags[..., 0], 0, np.abs(result.data[..., 0] - field) ** 2), axis=1
        )
    )
    reference = _delay_notch_residual(obj, geometry)
    recovered = result.recovered
    assert recovered.any()
    assert np.all(left[recovered] < 2 * reference[recovered] + 1)


def _rate_clean(data, is_pad, time_s, geometry, flags=None) -> SegmentFilterResult:
    return rate_clean_segment(
        data=data,
        flags=np.zeros(data.shape, bool) if flags is None else flags,
        is_pad=is_pad,
        time_s=time_s,
        freq_hz=FREQ_HZ,
        geometry=geometry,
        visibility_noise=0.1,
    )


def _rms(values: NDArray[np.complexfloating[Any]]) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


@pytest.mark.parametrize("object_amplitude", [1.0, 100.0, 1000.0])
def test_rate_clean_removes_object(object_amplitude: float) -> None:
    """Every candidate is recovered, the end rows included, and what is left of
    the object is at the noise (0.1 per part, 0.14 in amplitude)"""
    data, is_pad, time_s, geometry, _, field = _segment(
        n_candidates=120, n_pad=16, object_amplitude=object_amplitude
    )
    result = _rate_clean(data, is_pad, time_s, geometry)
    assert result.outcome == SegmentOutcome.FILTERED
    assert result.recovered is not None
    assert result.data is not None
    assert np.array_equal(result.recovered, ~is_pad)
    left = result.data[..., 0] - field
    assert _rms(left[~is_pad]) < 0.17
    candidates = np.flatnonzero(~is_pad)
    assert _rms(left[candidates[:8]]) < 0.2
    assert _rms(left[candidates[-8:]]) < 0.2


def test_rate_clean_keeps_field() -> None:
    """With no object present the field is unchanged"""
    data, is_pad, time_s, geometry, _, field = _segment(
        n_candidates=120, n_pad=16, object_amplitude=0.0, noise=0.0
    )
    result = _rate_clean(data, is_pad, time_s, geometry)
    assert result.data is not None
    np.testing.assert_allclose(result.data[..., 0], field, atol=1e-6)


def test_rate_clean_not_separable() -> None:
    data, is_pad, time_s, geometry, *_ = _segment(
        n_candidates=120, n_pad=16, object_rate=1e-13
    )
    result = _rate_clean(data, is_pad, time_s, geometry)
    assert result.outcome == SegmentOutcome.NOT_SEPARABLE


def test_rate_clean_with_random_flags() -> None:
    """Each channel's own time weights give it its own response, so scattered
    flags leave no more of the object behind"""
    data, is_pad, time_s, geometry, _, field = _segment(n_candidates=120, n_pad=16)
    flags = np.random.default_rng(1).random(data.shape) < 0.05
    data[flags] = 1e6
    result = _rate_clean(data, is_pad, time_s, geometry, flags=flags)
    assert result.recovered is not None
    assert result.data is not None
    assert np.array_equal(result.recovered, ~is_pad)
    left = (result.data[..., 0] - field)[~flags[..., 0] & ~is_pad[:, None]]
    assert _rms(left) < 0.17


def test_tukey_rate_filter_recovers_flagged_rows(tmp_path: Path) -> None:
    """Chunks from the delay notch, fed to the rate filter in row order,
    recover some of the rows the notch flags"""
    n_rows, chunk_rows = 120, 40
    time_s = T0_S + DT_S * np.arange(n_rows)
    delay_s = OBJECT_RATE * (time_s - time_s[n_rows // 2])
    phase = 2j * np.pi * FREQ_HZ[None, :]
    field = np.exp(phase * 3e-9) + 0.5 * np.exp(phase * -20e-9)
    vis = (100 * np.exp(phase * delay_s[:, None]) + field)[..., None]
    w_delays = WDelays(
        object_name="sun",
        w_delays=delay_s[None] * u.s,
        b_map={(0, 1): 0},
        time_map={t * u.s: i for i, t in enumerate(time_s)},
        elevation=np.full(n_rows, 45.0) * u.deg,
        guard_region=np.full((1, n_rows), 20e-9) * u.s,
        w_rates=np.full((1, n_rows), OBJECT_RATE) * u.dimensionless_unscaled,
        rate_guard_region=np.full((1, n_rows), 2e-13) * u.dimensionless_unscaled,
    )
    options = TukeyTractorOptions(rate_filter=True)
    rate_filter = make_rate_filter_run(
        settings=RateFilterSettings(
            w_delays_list=[w_delays],
            freq_hz=FREQ_HZ,
            segment_filter=partial(
                filter_segment,
                delay_width_s=options.outer_width_ns * 1e-9,
                delay_roll_off_s=options.tukey_width_ns * 1e-9,
                roll_off_rows=16,
            ),
            field_delay_s=options.outer_width_ns * 1e-9,
            elevation_cut_deg=options.elevation_cut_deg,
            ignore_nyquist_zone=options.ignore_nyquist_zone,
            plot_dir=tmp_path,
        ),
    )

    flagged, updates = [], []
    for start in range(0, n_rows, chunk_rows):
        rows = slice(start, start + chunk_rows)
        chunk = DataChunk(
            masked_data=np.ma.masked_array(
                vis[rows], mask=np.zeros(vis[rows].shape, bool)
            ),
            freq_chan=FREQ_HZ * u.Hz,
            time_mjds=time_s[rows],
            ant_1=np.zeros(chunk_rows, dtype=np.int64),
            ant_2=np.ones(chunk_rows, dtype=np.int64),
            row_start=start,
            chunk_size=chunk_rows,
        )
        result = compute_tukey_multi_taper(
            data_chunk=chunk, tukey_tractor_options=options, w_delays_list=[w_delays]
        )
        assert result.rate_payload is not None
        flagged.append(result.rate_payload.delay_contaminated)
        updates += update_rate_filter(rate_filter, result.rate_payload)
    updates += flush_rate_filter(rate_filter)

    flagged_rows = np.flatnonzero(np.concatenate(flagged))
    rows = np.concatenate([update.rows for update in updates])
    data = np.concatenate([update.data for update in updates])
    assert len(flagged_rows) > 0
    assert len(updates) > 1
    assert len(set(rows)) == len(rows)
    assert set(rows) <= set(flagged_rows)
    assert rate_filter.summary.recovered == len(rows) > len(flagged_rows) / 2
    left = np.sqrt(np.mean(np.abs(data[..., 0] - field) ** 2, axis=1))
    assert np.median(left) < 20
    assert not any(update.flags.any() for update in updates)
    assert rate_filter.summary.plotted == len(updates)
    assert (tmp_path / "rate_filter_0_1_0.png").exists()
