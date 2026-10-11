from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from jolly_roger.plots import plot_rate_filter_segment
from jolly_roger.rates import (
    ContaminatedSegment,
    RateFilterOutcome,
    RateFilterPayload,
    RateFilterResult,
    RateFilterSettings,
    RateFilterSummary,
    SegmentAccumulator,
    _axis_protection,
    _footprint_notch,
    _make_footprints,
    _SegmentGrid,
    compute_rate_contamination,
    flush_segment_accumulator,
    log_rate_filter_summary,
    rate_filter_segment,
    rate_filter_segments,
    record_rate_filter_result,
    update_segment_accumulator,
)
from jolly_roger.uvws import WDelays, get_w_rates

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

    payload = RateFilterPayload(
        rows=row_start + np.arange(n_rows),
        ant_1=np.zeros(n_rows, dtype=int),
        ant_2=ant_2,
        time_mjds=T0_S + time_idx * DT_S,
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
    keep_diagnostics: bool = False,
) -> tuple[RateFilterResult, NDArray[np.bool_]]:
    accumulator = SegmentAccumulator(pad_timesteps=pad_timesteps)
    released = _update(accumulator, recoverable, data=data, weights=weights)
    released += flush_segment_accumulator(accumulator)
    assert len(released) == 1

    result = rate_filter_segment(
        segment=released[0],
        freq_chan=FREQ_HZ * u.Hz,
        w_delays_list=[w_delays],
        settings=settings
        or RateFilterSettings(outer_width_ns=10.0, tukey_width_ns=5.0),
        keep_diagnostics=keep_diagnostics,
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

    assert result.outcome is RateFilterOutcome.FILTERED
    assert result.data is not None
    assert result.data.shape == data.shape
    assert _suppression_db(result, object_vis) < -10.0
    assert np.abs(np.mean(result.data) - 1.0) < 0.05
    assert 0.0 < result.nulled_fraction < 0.1


def test_rate_filter_wrong_sign_does_not_null() -> None:
    """Guard against the sign convention silently flipping"""
    data, object_vis, w_delays = _make_segment_inputs(sign=-1.0)
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))

    assert result.outcome is RateFilterOutcome.FILTERED
    assert _suppression_db(result, object_vis) > -1.0


def test_rate_filter_preserves_field() -> None:
    """Data only containing the field are not modified"""
    data, _, w_delays = _make_segment_inputs(object_amp=0.0)
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))

    assert result.outcome is RateFilterOutcome.FILTERED
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

    assert result.outcome is RateFilterOutcome.RATE_CONTAMINATED
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

    assert result.outcome is RateFilterOutcome.RATE_CONTAMINATED


def test_rate_filter_too_short() -> None:
    data, _, w_delays = _make_segment_inputs()
    recoverable = np.zeros(len(data), dtype=bool)
    recoverable[30:34] = True
    result, _ = _filter(data, w_delays, recoverable)

    assert result.outcome is RateFilterOutcome.TOO_SHORT
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

    assert result.outcome is RateFilterOutcome.NOTHING_TO_NULL


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
        assert result.outcome is RateFilterOutcome.RATE_CONTAMINATED
    else:
        assert result.outcome is RateFilterOutcome.FILTERED
        assert _suppression_db(result, object_vis) < -15.0


def test_rate_filter_summary() -> None:
    summary = RateFilterSummary()
    record_rate_filter_result(
        summary,
        RateFilterResult(rows=np.arange(4), outcome=RateFilterOutcome.FILTERED),
    )
    record_rate_filter_result(
        summary,
        RateFilterResult(rows=np.arange(2), outcome=RateFilterOutcome.TOO_SHORT),
    )

    assert summary.segments_filtered == 1
    assert summary.rows_filtered == 4
    assert summary.not_filtered["too short"] == 1
    assert summary.rows_not_filtered == 2
    log_rate_filter_summary(summary, doubly_contaminated_rows=3)


def test_rate_filter_diagnostics_only_when_requested() -> None:
    data, _, w_delays = _make_segment_inputs()
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))
    assert result.diagnostics is None


def test_rate_filter_diagnostics() -> None:
    data, _, w_delays = _make_segment_inputs(n_time=48)
    recoverable = np.zeros(len(data), dtype=bool)
    recoverable[10:40] = True
    result, _ = _filter(
        data, w_delays, recoverable, pad_timesteps=4, keep_diagnostics=True
    )

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert (diagnostics.ant_1, diagnostics.ant_2) == (0, 1)
    # The segment starts with its leading padding
    assert diagnostics.first_row == 6
    shape = (len(diagnostics.rate_hz), len(diagnostics.delay_s))
    assert diagnostics.before.shape == shape
    assert diagnostics.after.shape == shape
    # The object is nulled so there is less power after
    assert np.sum(diagnostics.after) < np.sum(diagnostics.before)

    (footprint,) = diagnostics.footprints
    assert footprint.object_name == "sun"
    assert footprint.sidelobe == 0
    assert len(footprint.delay_s) == len(footprint.rate_low_hz) == 38
    assert np.all(footprint.active)
    assert not np.any(footprint.all_rates)
    # 2e-11 s/s at the central frequency of 0.95 GHz
    np.testing.assert_allclose(
        (footprint.rate_low_hz + footprint.rate_high_hz) / 2, 2e-11 * 0.95e9
    )


def test_plot_rate_filter_segment(tmp_path: Path) -> None:
    data, _, w_delays = _make_segment_inputs()
    result, _ = _filter(
        data, w_delays, np.ones(len(data), dtype=bool), keep_diagnostics=True
    )
    assert result.diagnostics is not None

    output_path = plot_rate_filter_segment(
        diagnostics=result.diagnostics, output_path=tmp_path / "segment.png"
    )
    assert output_path.exists()


def _grid(n_time: int, delay_transition: float) -> _SegmentGrid:
    """A unitless delay-rate plane, with delays -100..100 and rates -50..50"""
    delay = np.linspace(-100.0, 100.0, 201)
    rate = np.linspace(-50.0, 50.0, 101)
    return _SegmentGrid(
        delay_s=delay,
        rate_hz=rate,
        time_s=np.arange(float(n_time)),
        max_delay_s=100.5,
        max_rate_hz=50.5,
        nu_min_hz=1.0,
        nu_max_hz=1.0,
        delay_transition_s=delay_transition,
    )


def _footprint(
    grid: _SegmentGrid,
    tau: NDArray[np.floating[Any]],
    tau_rate: NDArray[np.floating[Any]],
    outer_width: float,
    rate_margin: float,
    all_rates: NDArray[np.bool_] | None = None,
):
    (footprint,) = _make_footprints(
        object_name="sun",
        tau_s=tau,
        tau_rate=tau_rate,
        active=np.ones(len(tau), dtype=bool),
        all_rates=np.zeros(len(tau), dtype=bool) if all_rates is None else all_rates,
        grid=grid,
        settings=RateFilterSettings(
            outer_width_ns=outer_width * 1e9, tukey_width_ns=0.0
        ),
        rate_margin_hz=rate_margin,
    )
    return footprint


def test_footprint_notch_is_a_smooth_taper() -> None:
    """Zero across the object's footprint, one far away, with a 1 - cos
    transition between. A stationary object gives a box, as the delay taper."""
    grid = _grid(n_time=4, delay_transition=10.0)
    footprint = _footprint(
        grid,
        tau=np.full(4, 20.0),
        tau_rate=np.zeros(4),
        outer_width=30.0,
        rate_margin=10.0,
    )
    notch = _footprint_notch(footprint, grid)
    delay, rate = grid.delay_s, grid.rate_hz
    assert notch.shape == (len(rate), len(delay))

    # Along delay through the centre of the object in rate
    profile = notch[rate == 0.0][0]
    assert np.all(profile[np.abs(delay - 20.0) < 20.0] == 0.0)
    assert np.all(profile[np.abs(delay - 20.0) > 30.0] == 1.0)
    transition = (np.abs(delay - 20.0) > 20.0) & (np.abs(delay - 20.0) < 30.0)
    assert np.all((profile[transition] > 0.0) & (profile[transition] < 1.0))
    assert np.max(np.abs(np.diff(profile))) < 0.3
    # Far from the object in rate nothing is nulled
    assert np.all(notch[np.abs(rate) > 10.0] == 1.0)


def test_footprint_notch_follows_a_moving_object() -> None:
    """An object whose delay and fringe-rate both change is nulled along its
    path, not across the box bounding it"""
    n_time = 41
    grid = _grid(n_time=n_time, delay_transition=2.0)
    footprint = _footprint(
        grid,
        tau=np.linspace(-40.0, 40.0, n_time),
        tau_rate=np.linspace(-30.0, 30.0, n_time),
        outer_width=3.0,
        rate_margin=3.0,
    )
    notch = _footprint_notch(footprint, grid)
    delay, rate = grid.delay_s, grid.rate_hz
    on_path = notch[np.argmin(np.abs(rate - 0.0)), np.argmin(np.abs(delay - 0.0))]
    off_path = notch[np.argmin(np.abs(rate - 30.0)), np.argmin(np.abs(delay + 40.0))]
    assert on_path == 0.0
    # The opposite corner of the bounding box is not on the path
    assert off_path == 1.0


def test_axis_protection_is_a_smooth_window() -> None:
    """One across the field, falling to zero with a 1 - cos transition"""
    x = np.linspace(-100.0, 100.0, 201)
    protection = _axis_protection(x=x, half_width=10.0, tukey_width=10.0)

    assert np.all(protection[np.abs(x) <= 10.0] == 1.0)
    assert np.all(protection[np.abs(x) >= 20.0] == 0.0)
    transition = (np.abs(x) > 10.0) & (np.abs(x) < 20.0)
    assert np.all((protection[transition] > 0.0) & (protection[transition] < 1.0))
    assert np.max(np.abs(np.diff(protection))) < 0.3


@pytest.mark.parametrize("tukey_width_ns", [5.0, 0.0])
def test_rate_filter_taper_has_no_hard_edges(tukey_width_ns: float) -> None:
    """Applied to white noise the taper is recovered as after/before in delay-rate
    space. It should have no step from zero to one between neighbouring cells, even
    without a requested delay transition."""
    data, _, w_delays = _make_segment_inputs()
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape)
    result, _ = _filter(
        noise,
        w_delays,
        np.ones(len(data), dtype=bool),
        settings=RateFilterSettings(outer_width_ns=10.0, tukey_width_ns=tukey_width_ns),
        keep_diagnostics=True,
    )
    assert result.outcome is RateFilterOutcome.FILTERED
    assert result.diagnostics is not None
    taper = result.diagnostics.after / result.diagnostics.before

    assert np.min(taper) < 1e-6
    assert np.any((taper > 0.05) & (taper < 0.95))
    assert np.max(np.abs(np.diff(taper, axis=0))) < 0.75
    assert np.max(np.abs(np.diff(taper, axis=1))) < 0.75


def test_rate_filter_preserves_field_without_delay_transition() -> None:
    data, _, w_delays = _make_segment_inputs(object_amp=0.0)
    result, _ = _filter(
        data,
        w_delays,
        np.ones(len(data), dtype=bool),
        settings=RateFilterSettings(outer_width_ns=10.0, tukey_width_ns=0.0),
    )
    assert result.outcome is RateFilterOutcome.FILTERED
    assert result.data is not None
    np.testing.assert_allclose(result.data, data, atol=1e-10)


def _auto_settings(auto_sidelobes: int = 1) -> RateFilterSettings:
    return RateFilterSettings(
        outer_width_ns=10.0,
        tukey_width_ns=5.0,
        auto_width=True,
        auto_sidelobes=auto_sidelobes,
    )


def test_rate_filter_auto_width_long_segment() -> None:
    """A long segment uses the main lobe and requested sidelobes, (N+1)/T"""
    data, object_vis, w_delays = _make_segment_inputs(n_time=64)
    recoverable = np.ones(len(data), dtype=bool)
    duration_s = (len(data) - 1) * DT_S

    auto, _ = _filter(
        data, w_delays, recoverable, settings=_auto_settings(), keep_diagnostics=True
    )
    default, _ = _filter(data, w_delays, recoverable)

    assert auto.outcome is RateFilterOutcome.FILTERED
    assert auto.diagnostics is not None
    (footprint,) = auto.diagnostics.footprints
    assert footprint.rate_margin_hz == pytest.approx(2 / duration_s)
    assert (
        _suppression_db(auto, object_vis) < _suppression_db(default, object_vis) + 0.5
    )


def test_rate_filter_auto_width_fits_short_segment() -> None:
    """A fixed (N+1)/T margin overlaps the field on a short segment, but the
    fitted margin is reduced until the object is separable"""
    n_time = 24
    data, object_vis, w_delays = _make_segment_inputs(n_time=n_time)
    recoverable = np.ones(n_time, dtype=bool)
    duration_s = (n_time - 1) * DT_S

    fixed, _ = _filter(
        data,
        w_delays,
        recoverable,
        settings=RateFilterSettings(
            outer_width_ns=10.0, tukey_width_ns=5.0, width_hz=3 / duration_s
        ),
    )
    assert fixed.outcome is RateFilterOutcome.RATE_CONTAMINATED

    auto, _ = _filter(
        data,
        w_delays,
        recoverable,
        settings=_auto_settings(auto_sidelobes=2),
        keep_diagnostics=True,
    )
    assert auto.outcome is RateFilterOutcome.FILTERED
    assert auto.diagnostics is not None
    (footprint,) = auto.diagnostics.footprints
    assert footprint.rate_margin_hz < 3 / duration_s
    assert footprint.rate_margin_hz >= 1 / duration_s * (1 - 1e-9)
    assert _suppression_db(auto, object_vis) < -8.0


def test_rate_filter_auto_width_rate_contaminated() -> None:
    """An object without delay-rate can not be separated, even at the main lobe"""
    data, _, w_delays = _make_segment_inputs(tau_rate=1e-14)
    result, _ = _filter(
        data, w_delays, np.ones(len(data), dtype=bool), settings=_auto_settings()
    )
    assert result.outcome is RateFilterOutcome.RATE_CONTAMINATED


def test_rate_filter_fixed_width_unchanged() -> None:
    """Without auto sizing the margin is the requested width, or two rate bins"""
    data, _, w_delays = _make_segment_inputs(n_time=64)
    recoverable = np.ones(len(data), dtype=bool)
    rate_bin_hz = 1 / (len(data) * DT_S)

    default, _ = _filter(data, w_delays, recoverable, keep_diagnostics=True)
    requested, _ = _filter(
        data,
        w_delays,
        recoverable,
        settings=RateFilterSettings(
            outer_width_ns=10.0, tukey_width_ns=5.0, width_hz=0.004
        ),
        keep_diagnostics=True,
    )
    assert default.diagnostics is not None
    assert requested.diagnostics is not None
    assert default.diagnostics.footprints[0].rate_margin_hz == pytest.approx(
        2 * rate_bin_hz
    )
    assert requested.diagnostics.footprints[0].rate_margin_hz == pytest.approx(0.004)


def test_plot_rate_filter_segment_unmasked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filtered values are drawn as they are: zeros are at the bottom of the
    colour scale rather than masked, and the taper's roll-off is visible"""
    data, _, w_delays = _make_segment_inputs()
    result, _ = _filter(
        data, w_delays, np.ones(len(data), dtype=bool), keep_diagnostics=True
    )
    assert result.diagnostics is not None

    figures: list[plt.Figure] = []
    monkeypatch.setattr("jolly_roger.plots.plt.close", figures.append)
    plot_rate_filter_segment(
        diagnostics=result.diagnostics, output_path=tmp_path / "segment.png"
    )

    (figure,) = figures
    after_ax = next(ax for ax in figure.axes if ax.get_title() == "After")
    mesh = after_ax.collections[0]
    values = np.ma.getdata(mesh.get_array())

    assert not np.any(np.ma.getmaskarray(mesh.get_array()))
    assert np.all(np.isfinite(values))
    assert np.any(values == 0.0)
    # Zero is the bottom of the colour scale, not a "bad" value
    assert mesh.norm(0.0) == pytest.approx(0.0)
    # Some of the taper's roll-off lies in the linear part of the scale
    linthresh = mesh.norm.linthresh
    assert np.any((values > 0.0) & (values < linthresh))
    plt.close(figure)


def test_rate_filter_records_notches() -> None:
    """A filtered segment reports the region nulled for each object"""
    data, _, w_delays = _make_segment_inputs()
    result, _ = _filter(
        data, w_delays, np.ones(len(data), dtype=bool), keep_diagnostics=True
    )
    assert result.outcome is RateFilterOutcome.FILTERED
    assert result.diagnostics is not None
    assert result.notches == result.diagnostics.footprints
    assert result.notches[0].object_name == "sun"

    static_data, _, static_w_delays = _make_segment_inputs(tau_rate=1e-14)
    failed, _ = _filter(
        static_data, static_w_delays, np.ones(len(static_data), dtype=bool)
    )
    assert failed.outcome is RateFilterOutcome.RATE_CONTAMINATED
    assert failed.notches == []


def _make_moving_segment_inputs(
    tau_fn, rate_fn, n_time: int = 64, object_amp: float = 5.0
) -> tuple[NDArray[np.complexfloating[Any]], NDArray[np.complexfloating[Any]], WDelays]:
    """A field source and an object with a given delay and delay-rate as a
    function of time from the middle of the segment"""
    time_s = T0_S + np.arange(n_time) * DT_S
    time_from_middle = time_s - time_s.mean()
    tau_s = tau_fn(time_from_middle)
    object_vis = object_amp * np.exp(2j * np.pi * FREQ_HZ[None, :] * tau_s[:, None])
    data = np.repeat((1.0 + object_vis)[..., None], 2, axis=-1)
    w_delays = WDelays(
        object_name="sun",
        w_delays=(tau_s * u.s)[None, :],
        b_map={(0, 1): 0},
        time_map={t * u.s: idx for idx, t in enumerate(time_s)},
        elevation=np.full(n_time, 45.0) * u.deg,
        w_rates=rate_fn(time_from_middle)[None, :] * u.dimensionless_unscaled,
    )
    return data, np.repeat(object_vis[..., None], 2, axis=-1), w_delays


def test_rate_filter_footprint_smaller_than_bounding_box() -> None:
    """When the delay-rate changes across a segment the object is nulled along its
    path, a fraction of the box bounding it, while still suppressed"""
    a, b = 2e-11, 2e-14
    data, object_vis, w_delays = _make_moving_segment_inputs(
        tau_fn=lambda t: a * t + b * t**2, rate_fn=lambda t: a + 2 * b * t
    )
    result, _ = _filter(
        data, w_delays, np.ones(len(data), dtype=bool), keep_diagnostics=True
    )
    assert result.outcome is RateFilterOutcome.FILTERED
    assert result.diagnostics is not None

    tau_ns = w_delays.w_delays.to(u.ns).value[0]
    tau_rate = get_w_rates(w_delays).value[0]
    delay_ns = result.diagnostics.delay_s * 1e9
    rate_mhz = result.diagnostics.rate_hz * 1e3
    margin_mhz = result.diagnostics.footprints[0].rate_margin_hz * 1e3
    in_box_delay = (delay_ns >= tau_ns.min() - 10) & (delay_ns <= tau_ns.max() + 10)
    in_box_rate = (rate_mhz >= FREQ_HZ.min() * tau_rate.min() * 1e3 - margin_mhz) & (
        rate_mhz <= FREQ_HZ.max() * tau_rate.max() * 1e3 + margin_mhz
    )
    bounding_box_fraction = in_box_delay.mean() * in_box_rate.mean()

    assert result.nulled_fraction < 0.6 * bounding_box_fraction
    assert _suppression_db(result, object_vis) < -10.0
    # The applied taper is attached for plotting
    assert result.diagnostics.taper.shape == (len(rate_mhz), len(delay_ns))


def test_rate_filter_footprint_separable_where_box_is_not() -> None:
    """The object passes through zero delay only while its fringe-rate is high, and
    through zero fringe-rate only far from zero delay. Its bounding box meets the
    field but its path does not, so it is filtered."""
    half_duration_s = 32 * DT_S
    data, _, w_delays = _make_moving_segment_inputs(
        tau_fn=lambda t: 60e-9 * t / half_duration_s,
        rate_fn=lambda t: 3.2e-11 * (1 - (t / half_duration_s) ** 2),
    )
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))
    assert result.outcome is RateFilterOutcome.FILTERED


def test_rate_filter_beyond_rate_nyquist_zone_contaminating() -> None:
    """An object beyond the Nyquist zone in fringe-rate can not be separated from
    the field in delay-rate, so where it crosses the field the segment remains
    flagged rather than being passed through"""
    data, _, w_delays = _make_moving_segment_inputs(
        tau_fn=lambda t: 3e-10 * t, rate_fn=lambda t: np.full_like(t, 3e-10)
    )
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))
    assert result.outcome is RateFilterOutcome.RATE_CONTAMINATED
    assert result.data is None
    assert result.notches == []


def test_plot_rate_filter_segment_wraps_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An object aliased beyond the edge of the panel in fringe-rate has its track
    and notch outline drawn inside the panel"""
    data, _, w_delays = _make_moving_segment_inputs(
        tau_fn=lambda t: 6.5e-11 * t, rate_fn=lambda t: np.full_like(t, 6.5e-11)
    )
    result, _ = _filter(
        data, w_delays, np.ones(len(data), dtype=bool), keep_diagnostics=True
    )
    assert result.outcome is RateFilterOutcome.FILTERED
    assert result.diagnostics is not None
    # ~62 mHz at the central frequency, beyond the 50 mHz edge
    (footprint,) = result.diagnostics.footprints
    assert np.all(np.abs(footprint.rate_low_hz + footprint.rate_high_hz) / 2 > 0.05)

    figures: list[plt.Figure] = []
    monkeypatch.setattr("jolly_roger.plots.plt.close", figures.append)
    plot_rate_filter_segment(
        diagnostics=result.diagnostics, output_path=tmp_path / "segment.png"
    )
    (figure,) = figures
    after_ax = next(ax for ax in figure.axes if ax.get_title() == "After")
    rate_mhz = result.diagnostics.rate_hz * 1e3

    track = [line for line in after_ax.lines if line.get_marker() == "."]
    assert sum(len(line.get_ydata()) for line in track) == len(data)
    for line in track:
        y = np.asarray(line.get_ydata())
        assert np.all((y >= rate_mhz.min() - 1e-9) & (y <= rate_mhz.max() + 1e-9))
    assert "Notch" in figure.axes[0].get_legend_handles_labels()[1]
    plt.close(figure)

    # The object's band starts at ~52 mHz, aliased to ~-48 mHz, so the taper must
    # reach across the edge of the axis (which wraps at +/- 50 mHz, not +48.4 mHz)
    taper = result.diagnostics.taper
    delay_col = np.argmin(np.abs(result.diagnostics.delay_s))
    assert taper[np.argmin(np.abs(rate_mhz + 48.4375)), delay_col] < 0.1


def _segment(data: NDArray[np.complexfloating[Any]]) -> ContaminatedSegment:
    """A single segment at the times of ``_times``, holding every row as contaminated"""
    accumulator = SegmentAccumulator()
    released = _update(
        accumulator, recoverable=np.ones(len(data), dtype=bool), data=data
    )
    (segment,) = released + flush_segment_accumulator(accumulator)
    return segment


def _object(
    name: str,
    time_s: NDArray[np.floating[Any]],
    tau_s: NDArray[np.floating[Any]],
    tau_rate: NDArray[np.floating[Any]],
    elevation_deg: NDArray[np.floating[Any]] | None = None,
) -> tuple[NDArray[np.complexfloating[Any]], WDelays]:
    """The visibilities of an object, and its predicted delays and rates"""
    n_time = len(time_s)
    w_delays = WDelays(
        object_name=name,
        w_delays=(tau_s * u.s)[None, :],
        b_map={(0, 1): 0},
        time_map={t * u.s: idx for idx, t in enumerate(time_s)},
        elevation=(np.full(n_time, 45.0) if elevation_deg is None else elevation_deg)
        * u.deg,
        w_rates=np.broadcast_to(tau_rate, (n_time,))[None, :]
        * u.dimensionless_unscaled,
    )
    vis = 5.0 * np.exp(2j * np.pi * FREQ_HZ[None, :] * tau_s[:, None])
    return np.repeat(vis[..., None], 2, axis=-1), w_delays


def _times(n_time: int) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    time_s = T0_S + np.arange(n_time) * DT_S
    return time_s, time_s - time_s.mean()


def test_compute_rate_contamination() -> None:
    """Rows are classified by the object's fringe-rate band against the field's,
    independent of the resolution of any segment"""
    n_time = 4
    time_s, _ = _times(n_time)
    settings = RateFilterSettings(
        outer_width_ns=10.0, tukey_width_ns=10.0, integration_s=DT_S
    )
    # A narrow band, so the rate Nyquist zone alone decides at high rates
    freq_chan = np.linspace(1.0, 1.05, 8) * u.GHz
    idx = np.zeros(n_time, dtype=int), np.arange(n_time)

    def _contaminated(rate: float, **kwargs) -> NDArray[np.bool_]:
        _, w_delays = _object("sun", time_s, np.zeros(n_time), np.full(n_time, rate))
        return compute_rate_contamination(
            w_delays=w_delays,
            baseline_idx=idx[0],
            time_idx=idx[1],
            freq_chan=freq_chan,
            settings=replace(settings, **kwargs),
        )

    # No delay-rate can not be separated, a moving object can
    assert np.all(_contaminated(0.0))
    assert not np.any(_contaminated(2e-11))
    # Within an absolute guard (2e-11 * 1 GHz = 20 mHz)
    assert np.all(_contaminated(2e-11, guard_hz=0.03))
    # Beyond the second fringe-rate Nyquist zone (1.5 / (2 * 10 s) = 75 mHz) the
    # object is contaminated, unless there is no limit
    assert np.all(_contaminated(1.7e-10))
    assert not np.any(_contaminated(1.7e-10, ignore_rate_nyquist_zone=None))

    with pytest.raises(ValueError, match="integration"):
        _contaminated(2e-11, integration_s=None)


def test_rate_filter_segments_split_at_contaminated_timesteps() -> None:
    """Timesteps where the object meets the field remain flagged, and the rest of
    the segment is filtered around them"""
    n_time = 64
    time_s, _ = _times(n_time)
    tau_rate = np.full(n_time, 3e-11)
    tau_rate[30:34] = 1e-14
    object_vis, w_delays = _object("sun", time_s, np.full(n_time, 1e-9), tau_rate)
    data = 1.0 + object_vis

    results = rate_filter_segments(
        segment=_segment(data),
        freq_chan=FREQ_HZ * u.Hz,
        w_delays_list=[w_delays],
        settings=RateFilterSettings(outer_width_ns=10.0, tukey_width_ns=5.0),
    )
    outcomes = {
        r.outcome: r.rows.tolist()
        for r in results
        if r.outcome is not RateFilterOutcome.FILTERED
    }
    filtered = [
        r.rows.tolist() for r in results if r.outcome is RateFilterOutcome.FILTERED
    ]

    assert outcomes == {RateFilterOutcome.RATE_CONTAMINATED: [30, 31, 32, 33]}
    assert filtered == [list(range(30)), list(range(34, 64))]
    # Every core row is in exactly one result
    rows = np.concatenate([r.rows for r in results])
    assert sorted(rows.tolist()) == list(range(n_time))


def test_rate_filter_inactive_times_do_not_contaminate() -> None:
    """Below the elevation cut the object is not nulled, as for the delay taper,
    so it does not need to be separable from the field there"""
    n_time = 64
    time_s, _ = _times(n_time)
    tau_rate = np.full(n_time, 3e-11)
    tau_rate[:20] = 1e-14
    elevation = np.where(np.arange(n_time) < 20, -10.0, 45.0)
    object_vis, w_delays = _object(
        "sun", time_s, np.full(n_time, 1e-9), tau_rate, elevation_deg=elevation
    )

    (result,) = rate_filter_segments(
        segment=_segment(1.0 + object_vis),
        freq_chan=FREQ_HZ * u.Hz,
        w_delays_list=[w_delays],
        settings=RateFilterSettings(outer_width_ns=10.0, tukey_width_ns=5.0),
    )
    assert result.outcome is RateFilterOutcome.FILTERED
    (footprint,) = result.notches
    np.testing.assert_array_equal(footprint.active, elevation >= -1.0)


def test_rate_filter_beyond_rate_zone_nulled_in_delay_only() -> None:
    """An object beyond the fringe-rate Nyquist zone, but clear of the field in
    delay, is nulled across all fringe-rates as the delay taper would"""
    n_time = 16
    time_s, from_middle = _times(n_time)
    vis_a, crossing = _object("A", time_s, 4e-11 * from_middle, np.full(n_time, 4e-11))
    vis_b, fast = _object(
        "B", time_s, 50e-9 + 2.5e-10 * from_middle, np.full(n_time, 2.5e-10)
    )

    (result,) = rate_filter_segments(
        segment=_segment(1.0 + vis_a + vis_b),
        freq_chan=FREQ_HZ * u.Hz,
        w_delays_list=[crossing, fast],
        settings=RateFilterSettings(outer_width_ns=10.0, tukey_width_ns=5.0),
    )
    assert result.outcome is RateFilterOutcome.FILTERED
    footprints = {fp.object_name: fp for fp in result.notches}
    assert not np.any(footprints["A"].all_rates)
    assert np.all(footprints["B"].all_rates)

    assert result.data is not None
    residual = result.data - 1.0
    objects = vis_a + vis_b
    assert (
        10 * np.log10(np.sum(np.abs(residual) ** 2) / np.sum(np.abs(objects) ** 2))
        < -10.0
    )


def test_rate_filter_sidelobe_footprints() -> None:
    """Delay sidelobes are nulled as the delay taper places them"""
    data, _, w_delays = _make_segment_inputs()
    result, _ = _filter(
        data,
        w_delays,
        np.ones(len(data), dtype=bool),
        settings=RateFilterSettings(
            outer_width_ns=4.0, tukey_width_ns=0.0, nth_sidelobe_null=2
        ),
    )
    assert result.outcome is RateFilterOutcome.FILTERED
    main_lobe = next(fp for fp in result.notches if fp.sidelobe == 0)
    offsets_ns = {
        fp.sidelobe: float(np.mean(fp.delay_s - main_lobe.delay_s)) * 1e9
        for fp in result.notches
    }
    assert offsets_ns == pytest.approx({0: 0.0, 1: 6.0, 2: 10.0, -1: -6.0, -2: -10.0})


def test_rate_filter_peak_shift_follows_an_offset_object() -> None:
    """An object offset in delay from its prediction (e.g. by the ionosphere) is
    found by the peak search, and nulled as well as when it is not offset"""
    n_time = 64
    time_s, from_middle = _times(n_time)
    delay_bin_s = 1 / (N_CHAN * np.mean(np.diff(FREQ_HZ)))
    predicted = 2e-11 * from_middle
    _, w_delays = _object("sun", time_s, predicted, 2e-11)
    settings = RateFilterSettings(outer_width_ns=10.0, tukey_width_ns=5.0)
    searched = replace(
        settings, peak_shift_search=True, peak_shift_search_width_ns=30.0
    )

    def _filter_offset(
        offset_bins: int, settings: RateFilterSettings
    ) -> tuple[float, float]:
        """The shift found, in delay bins, and the suppression of the object"""
        object_vis, _ = _object(
            "sun", time_s, predicted + offset_bins * delay_bin_s, 2e-11
        )
        result, _ = _filter(
            1.0 + object_vis, w_delays, np.ones(n_time, dtype=bool), settings=settings
        )
        assert result.outcome is RateFilterOutcome.FILTERED
        (footprint,) = result.notches
        shift_bins = float(np.mean(footprint.delay_s - predicted) / delay_bin_s)
        return shift_bins, _suppression_db(result, object_vis)

    shift_bins, unshifted_db = _filter_offset(offset_bins=0, settings=searched)
    assert shift_bins == pytest.approx(0.0)
    _, missed_db = _filter_offset(offset_bins=3, settings=settings)
    shift_bins, found_db = _filter_offset(offset_bins=3, settings=searched)
    assert shift_bins == pytest.approx(3.0)
    assert found_db == pytest.approx(unshifted_db, abs=0.5)
    assert found_db < missed_db - 3.0


def test_footprint_notch_nulls_the_band_edge_at_small_margins() -> None:
    """With a margin narrower than the transition the zero region still covers the
    object's whole fringe-rate band"""
    grid = _grid(n_time=4, delay_transition=10.0)
    footprint = _footprint(
        grid,
        tau=np.zeros(4),
        tau_rate=np.full(4, 15.0),
        outer_width=10.0,
        rate_margin=1.0,
    )
    footprint = replace(
        footprint, rate_low_hz=np.full(4, 10.0), rate_high_hz=np.full(4, 20.0)
    )
    notch = _footprint_notch(footprint, grid)
    in_band = (grid.rate_hz >= 10.0) & (grid.rate_hz <= 20.0)
    assert np.all(notch[in_band, np.argmin(np.abs(grid.delay_s))] == 0.0)


def test_rate_filter_unflagged_nan_only_affects_its_row() -> None:
    """A non-finite visibility that is not flagged is excluded from the transform,
    so the rest of the segment is filtered and only its own sample is flagged"""
    data, object_vis, w_delays = _make_segment_inputs()
    data = data.copy()
    data[10, 5, 0] = np.nan
    result, _ = _filter(data, w_delays, np.ones(len(data), dtype=bool))

    assert result.outcome is RateFilterOutcome.FILTERED
    assert result.data is not None
    assert result.flags is not None
    assert np.all(np.isfinite(result.data))
    assert result.flags[10, 5, 0]
    assert np.sum(result.flags) == 1
    assert _suppression_db(result, object_vis) < -10.0


def test_accumulator_gaps_measured_in_time() -> None:
    """With a maximum gap in time, a scan gap splits a segment, but time indices
    that jump because centroids differ between baselines do not"""
    accumulator = SegmentAccumulator(max_time_gap_s=1.5 * DT_S)
    n_rows = 6
    # Each integration has several centroids, so this baseline's indices jump
    time_idx = np.arange(n_rows) * 12
    time_s = T0_S + np.arange(n_rows) * DT_S + 0.01 * np.arange(n_rows)
    time_s[4:] += 30 * 60.0
    payload = RateFilterPayload(
        rows=np.arange(n_rows),
        ant_1=np.zeros(n_rows, dtype=int),
        ant_2=np.ones(n_rows, dtype=int),
        time_mjds=time_s,
        baseline_idx=np.zeros(n_rows, dtype=int),
        time_idx=time_idx,
        original_masked_data=np.ma.masked_array(np.ones((n_rows, 4, 2), dtype=complex)),
        delay_contaminated=np.ones(n_rows, dtype=bool),
        recoverable=np.ones(n_rows, dtype=bool),
    )
    released = update_segment_accumulator(accumulator=accumulator, payload=payload)
    released += flush_segment_accumulator(accumulator)
    assert [seg.rows for seg in released] == [[0, 1, 2, 3], [4, 5]]


def test_rate_filter_nulls_the_band_clear_of_the_field_protection() -> None:
    """Whenever a segment is filtered the object's fringe-rate band is fully nulled,
    so the field's protection, which rolls off beyond the field, never reaches it.
    The margin here is narrower than the transition of the taper."""
    n_time = 64
    time_s, _ = _times(n_time)
    rate_bin_hz = 1 / (n_time * DT_S)
    settings = RateFilterSettings(
        outer_width_ns=10.0, tukey_width_ns=5.0, width_hz=0.5 * rate_bin_hz
    )
    outcomes = set()
    for near_edge_bins in np.arange(1.2, 4.0, 0.2):
        tau_rate = near_edge_bins * rate_bin_hz / FREQ_HZ.min()
        object_vis, w_delays = _object(
            "sun", time_s, np.full(n_time, 1e-9), np.full(n_time, tau_rate)
        )
        result = rate_filter_segment(
            segment=_segment(1.0 + object_vis),
            freq_chan=FREQ_HZ * u.Hz,
            w_delays_list=[w_delays],
            settings=settings,
            keep_diagnostics=True,
        )
        outcomes.add(result.outcome)
        if result.outcome is not RateFilterOutcome.FILTERED:
            continue
        assert result.diagnostics is not None
        rate_hz = result.diagnostics.rate_hz
        in_band = (rate_hz >= FREQ_HZ.min() * tau_rate) & (
            rate_hz <= FREQ_HZ.max() * tau_rate
        )
        delay_col = np.argmin(np.abs(result.diagnostics.delay_s - 1e-9))
        assert np.any(in_band)
        np.testing.assert_array_equal(result.diagnostics.taper[in_band, delay_col], 0.0)
    assert outcomes == {RateFilterOutcome.FILTERED, RateFilterOutcome.RATE_CONTAMINATED}
