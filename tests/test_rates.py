from __future__ import annotations

from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pytest
from numpy.typing import NDArray

from jolly_roger.plots import plot_rate_filter_segment
from jolly_roger.rates import (
    RateFilterResult,
    RateFilterSettings,
    RateFilterSummary,
    SegmentAccumulator,
    _axis_notch,
    _axis_protection,
    flush_segment_accumulator,
    log_rate_filter_summary,
    rate_filter_segment,
    record_rate_filter_result,
    update_segment_accumulator,
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

    return update_segment_accumulator(
        accumulator=accumulator,
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
    record_rate_filter_result(
        summary, RateFilterResult(rows=np.arange(4), success=True)
    )
    record_rate_filter_result(
        summary, RateFilterResult(rows=np.arange(2), success=False, reason="too short")
    )

    assert summary.segments_filtered == 1
    assert summary.rows_filtered == 4
    assert summary.failures["too short"] == 1
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

    (track,) = diagnostics.tracks
    assert track.object_name == "sun"
    assert len(track.delay_s) == len(track.rate_hz) == 38
    # 2e-11 s/s at the central frequency of 0.95 GHz
    np.testing.assert_allclose(track.rate_hz, 2e-11 * 0.95e9)


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


def test_axis_notch_is_a_smooth_taper() -> None:
    """Zero at the object, one far away, with a 1 - cos transition between"""
    x = np.linspace(-100.0, 100.0, 201)
    notch = _axis_notch(x=x, center=20.0, outer_width=30.0, tukey_width=10.0)

    assert notch[x == 20.0] == 0.0
    assert np.all(notch[np.abs(x - 20.0) < 20.0] == 0.0)
    assert np.all(notch[np.abs(x - 20.0) > 30.0] == 1.0)
    transition = (np.abs(x - 20.0) > 20.0) & (np.abs(x - 20.0) < 30.0)
    assert np.all((notch[transition] > 0.0) & (notch[transition] < 1.0))
    assert np.max(np.abs(np.diff(notch))) < 0.3


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
    assert result.success
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
    assert result.success
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

    assert auto.success
    assert auto.diagnostics is not None
    (track,) = auto.diagnostics.tracks
    assert track.rate_width_hz == pytest.approx(2 / duration_s)
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
    assert not fixed.success
    assert fixed.reason == "rate-contaminated"

    auto, _ = _filter(
        data,
        w_delays,
        recoverable,
        settings=_auto_settings(auto_sidelobes=2),
        keep_diagnostics=True,
    )
    assert auto.success
    assert auto.diagnostics is not None
    (track,) = auto.diagnostics.tracks
    assert track.rate_width_hz < 3 / duration_s
    assert track.rate_width_hz >= 1 / duration_s * (1 - 1e-9)
    assert _suppression_db(auto, object_vis) < -8.0


def test_rate_filter_auto_width_rate_contaminated() -> None:
    """An object without delay-rate can not be separated, even at the main lobe"""
    data, _, w_delays = _make_segment_inputs(tau_rate=1e-14)
    result, _ = _filter(
        data, w_delays, np.ones(len(data), dtype=bool), settings=_auto_settings()
    )
    assert not result.success
    assert result.reason == "rate-contaminated"


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
    assert default.diagnostics.tracks[0].rate_width_hz == pytest.approx(2 * rate_bin_hz)
    assert requested.diagnostics.tracks[0].rate_width_hz == pytest.approx(0.004)
