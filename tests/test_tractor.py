"""Tests around the tractor'ing. Some are simple, some are complex, but
all are important in their own way"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from casacore.tables import table
from numpy import ma

from jolly_roger.baselines import get_open_ms_tables
from jolly_roger.rates import (
    RateFilterResult,
    SegmentAccumulator,
    update_segment_accumulator,
)
from jolly_roger.tractor import (
    DataChunk,
    RateFilterProcessor,
    RateFilterWriteBuffer,
    TukeyTractorOptions,
    _rate_filter_settings,
    add_to_rate_filter_write_buffer,
    apply_roll_for_taper,
    compute_auto_taper_widths,
    compute_rate_contamination,
    compute_tukey_multi_taper,
    find_idx_of_closest_delay,
    finish_rate_filter,
    flush_rate_filter_write_buffer,
    make_rate_filter_plots,
    make_search_window,
    merge_rate_filter_results,
    tukey_tractor,
    write_rate_filtered_segment,
)
from jolly_roger.uvws import WDelays


def test_make_search_window() -> None:
    """Perform a very simple check to ensure that the window function performs
    correctly"""
    times = (np.arange(200) - 100) * 1e-9 * u.s
    width_ns = 10

    mask = make_search_window(x=times, width_ns=width_ns)
    assert np.sum(mask) == 21
    assert np.all(~mask[:90])
    assert np.all(mask[90 : 90 + 21])
    assert np.all(~mask[90 + 21 :])


def test_apply_roll_for_taper() -> None:
    """Ensure that a roll can be made over the delay time
    axis, where the taper shape is (row, demay_time)"""

    base = np.array(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]
    )
    shifts = np.array([0, 1, 2, 3])
    shifted = np.array(
        [[1, 2, 3, 4, 5], [10, 6, 7, 8, 9], [14, 15, 11, 12, 13], [18, 19, 20, 16, 17]]
    )

    rolled = apply_roll_for_taper(taper=base, shifts=shifts)
    assert np.all(rolled == shifted)


def test_find_idx_of_closest_delay() -> None:
    """Given a delay domain, confirm that the correct index is returned
    when attempting to find the closest"""
    x = np.linspace(-100, 100, 200) * u.s
    object = np.array((-90, 90)) * u.s

    idxs = find_idx_of_closest_delay(x=x, object_delays=object)
    assert len(idxs) == 2
    assert idxs[0] == 10
    assert idxs[1] == 189


def test_tractor_run1_with_peak_search_and_width(ms_example) -> None:
    """A very simple end-to-end test identifying crashes. This
    invokes the peak search mode"""

    new_column = "JACKS_DATA"

    tukey_tractor_options = TukeyTractorOptions(
        auto_size=False,
        output_column=new_column,
        peak_shift_search=True,
        peak_shift_search_width_ns=20,
        ignore_nyquist_zone=1000,
        elevation_cut_deg=-100,
        tukey_width_ns=20,
        outer_width_ns=30,
    )
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column not in cols

    tractor_results = tukey_tractor(
        ms_path=Path(ms_example),
        tukey_tractor_options=tukey_tractor_options,
    )

    assert tractor_results.output_plots is None
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column in cols


def test_tractor_run1_with_peak_search(ms_example) -> None:
    """A very simple end-to-end test identifying crashes. This
    invokes the peak search mode"""

    new_column = "JACKS_DATA"

    tukey_tractor_options = TukeyTractorOptions(
        auto_size=False,
        output_column=new_column,
        peak_shift_search=True,
        ignore_nyquist_zone=1000,
        elevation_cut_deg=-100,
        tukey_width_ns=20,
        outer_width_ns=30,
    )
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column not in cols

    tractor_results = tukey_tractor(
        ms_path=Path(ms_example),
        tukey_tractor_options=tukey_tractor_options,
    )

    assert tractor_results.output_plots is None
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column in cols


def test_tractor_run1(ms_example) -> None:
    """A very simple end-to-end test identifying crashes"""

    new_column = "JACKS_DATA"

    tukey_tractor_options = TukeyTractorOptions(
        auto_size=True,
        guard_field=True,
        object_minimum_flux=0.05,
        output_column=new_column,
    )
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column not in cols

    tractor_results = tukey_tractor(
        ms_path=Path(ms_example),
        tukey_tractor_options=tukey_tractor_options,
    )

    assert tractor_results.output_plots is None
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column in cols


def test_tractor_run2(ms_example) -> None:
    """A very simple end-to-end test identifying crashes"""

    new_column = "JACKS_DATA"

    tukey_tractor_options = TukeyTractorOptions(
        auto_size=True,
        guard_field=True,
        object_minimum_flux=0.05,
        output_column=new_column,
        make_plots=True,
        number_of_plots=1,
    )
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column not in cols

    tractor_results = tukey_tractor(
        ms_path=Path(ms_example),
        tukey_tractor_options=tukey_tractor_options,
    )

    assert tractor_results.output_plots is not None
    assert len(tractor_results.output_plots) == 1

    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column in cols


def _make_data_chunk(n_time: int = 8, n_chan: int = 16, n_pol: int = 2) -> DataChunk:
    """A synthetic single-baseline data chunk. No pirates were harmed."""
    rng = np.random.default_rng(1934)
    data = rng.standard_normal((n_time, n_chan, n_pol)) + 1j * rng.standard_normal(
        (n_time, n_chan, n_pol)
    )
    return DataChunk(
        masked_data=ma.masked_array(data, mask=np.zeros_like(data, dtype=bool)),
        freq_chan=np.linspace(1.0, 2.0, n_chan) * u.GHz,
        phase_center=SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg),
        uvws_phase_center=np.zeros((n_time, 3)) * u.m,
        time_mjds=np.arange(n_time, dtype=float),
        ant_1=np.zeros(n_time, dtype=np.int64),
        ant_2=np.ones(n_time, dtype=np.int64),
        row_start=0,
        chunk_size=n_time,
    )


def _make_w_delays(n_time: int, elevation_deg: float = 45.0) -> WDelays:
    """A synthetic single-baseline WDelays, matching `_make_data_chunk`'s indexing."""
    return WDelays(
        object_name="sun",
        w_delays=np.zeros((1, n_time)) * u.s,
        b_map={(0, 1): 0},
        time_map={t * u.s: idx for idx, t in enumerate(np.arange(n_time, dtype=float))},
        elevation=np.full(n_time, elevation_deg) * u.deg,
    )


def test_compute_tukey_multi_taper_applies_taper() -> None:
    """Array-only test of the tractor's compute core: a taper is applied
    to a synthetic DataChunk+WDelays with no measurement set in sight."""
    n_time = 8
    data_chunk = _make_data_chunk(n_time=n_time)
    w_delays = _make_w_delays(n_time=n_time)

    result = compute_tukey_multi_taper(
        data_chunk=data_chunk,
        tukey_tractor_options=TukeyTractorOptions(),
        w_delays_list=[w_delays],
    )

    assert not result.nothing_to_do
    assert result.update_data
    assert result.data_chunk is not None
    assert result.data_chunk.masked_data.shape == data_chunk.masked_data.shape


def test_compute_tukey_multi_taper_skips_below_elevation_cut() -> None:
    """When the target is below the elevation cut for the whole chunk, nothing
    should be tapered."""
    n_time = 8
    data_chunk = _make_data_chunk(n_time=n_time)
    w_delays = _make_w_delays(n_time=n_time, elevation_deg=-10.0)

    result = compute_tukey_multi_taper(
        data_chunk=data_chunk,
        tukey_tractor_options=TukeyTractorOptions(),
        w_delays_list=[w_delays],
    )

    assert result.nothing_to_do
    assert result.data_chunk is data_chunk


def _with_rates(w_delays: WDelays, rate: float, guard: float | None = None) -> WDelays:
    """Attach a constant delay-rate (and rate guard) to a WDelays"""
    shape = w_delays.w_delays.shape
    return WDelays(
        object_name=w_delays.object_name,
        w_delays=w_delays.w_delays,
        b_map=w_delays.b_map,
        time_map=w_delays.time_map,
        elevation=w_delays.elevation,
        w_rates=np.full(shape, rate) * u.dimensionless_unscaled,
        rate_guard_region=None
        if guard is None
        else np.full(shape, guard) * u.dimensionless_unscaled,
    )


def test_compute_rate_contamination() -> None:
    n_time = 4
    w_delays = _make_w_delays(n_time=n_time)
    freq_chan = np.linspace(1.0, 2.0, 16) * u.GHz
    idx = np.zeros(n_time, dtype=int), np.arange(n_time)

    def _contaminated(w: WDelays, **kwargs) -> np.ndarray:
        return compute_rate_contamination(
            w_delays=w,
            baseline_idx=idx[0],
            time_idx=idx[1],
            freq_chan=freq_chan,
            tukey_tractor_options=TukeyTractorOptions(rate_filter=True, **kwargs),
        )

    # No rate (derived from constant delays) can not be separated
    assert np.all(_contaminated(w_delays))
    # Moving object, no guard
    assert not np.any(_contaminated(_with_rates(w_delays, rate=1e-11)))
    # Moving object, within the field guard
    assert np.all(_contaminated(_with_rates(w_delays, rate=1e-11, guard=2e-11)))
    # Moving object, within the absolute guard (1e-11 * 1 GHz = 0.01 Hz)
    assert np.all(
        _contaminated(_with_rates(w_delays, rate=1e-11), rate_filter_guard_hz=0.02)
    )
    assert not np.any(
        _contaminated(_with_rates(w_delays, rate=1e-11), rate_filter_guard_hz=0.005)
    )


def test_compute_tukey_multi_taper_rate_payload() -> None:
    """The object sits at delay 0 so every row is contaminated in delay. Whether
    it is recoverable depends on its delay-rate."""
    n_time = 8
    data_chunk = _make_data_chunk(n_time=n_time)
    original = data_chunk.masked_data
    original_values = original.data.copy()

    static = compute_tukey_multi_taper(
        data_chunk=data_chunk,
        tukey_tractor_options=TukeyTractorOptions(rate_filter=True),
        w_delays_list=[_make_w_delays(n_time=n_time)],
    )
    payload = static.rate_payload
    assert payload is not None
    assert np.all(payload.delay_contaminated)
    assert not np.any(payload.recoverable)
    # The original visibilities are kept, untapered
    assert payload.original_masked_data is original
    np.testing.assert_array_equal(payload.original_masked_data.data, original_values)
    assert static.data_chunk is not None
    assert not np.allclose(static.data_chunk.masked_data.data, original_values)
    np.testing.assert_array_equal(payload.time_idx, np.arange(n_time))
    np.testing.assert_array_equal(payload.baseline_idx, np.zeros(n_time))

    moving = compute_tukey_multi_taper(
        data_chunk=_make_data_chunk(n_time=n_time),
        tukey_tractor_options=TukeyTractorOptions(rate_filter=True),
        w_delays_list=[_with_rates(_make_w_delays(n_time=n_time), rate=1e-11)],
    )
    assert moving.rate_payload is not None
    assert np.all(moving.rate_payload.recoverable)


def test_compute_tukey_multi_taper_no_rate_payload_by_default() -> None:
    n_time = 8
    result = compute_tukey_multi_taper(
        data_chunk=_make_data_chunk(n_time=n_time),
        tukey_tractor_options=TukeyTractorOptions(),
        w_delays_list=[_make_w_delays(n_time=n_time)],
    )
    assert result.rate_payload is None


def test_compute_tukey_multi_taper_rate_payload_nothing_to_do() -> None:
    """Rows are still reported (as clean) when there is nothing to taper"""
    n_time = 8
    result = compute_tukey_multi_taper(
        data_chunk=_make_data_chunk(n_time=n_time),
        tukey_tractor_options=TukeyTractorOptions(rate_filter=True),
        w_delays_list=[_make_w_delays(n_time=n_time, elevation_deg=-10.0)],
    )
    assert result.nothing_to_do
    assert result.rate_payload is not None
    assert not np.any(result.rate_payload.delay_contaminated)
    assert not np.any(result.rate_payload.recoverable)


@pytest.mark.parametrize("pad", [0, 2])
def test_tractor_rate_filter(
    ms_example, monkeypatch: pytest.MonkeyPatch, caplog, pad: int
) -> None:
    """End-to-end run identifying crashes when delay-rate filtering. The target
    is placed close to the phase direction so that it enters the delay
    contaminated zone on short baselines."""
    with table(str(ms_example / "FIELD"), ack=False) as tab:
        phase_dir = tab.getcol("PHASE_DIR")[0, 0]
    phase = SkyCoord(*phase_dir, unit="rad")
    target = SkyCoord(phase.ra, phase.dec + 8 * u.deg)
    monkeypatch.setattr(SkyCoord, "from_name", staticmethod(lambda _: target))

    new_column = "JACKS_DATA"
    tukey_tractor_options = TukeyTractorOptions(
        target_objects=("NEAR_FIELD",),
        outer_width_ns=4.0,
        tukey_width_ns=2.0,
        guard_field=True,
        output_column=new_column,
        rate_filter=True,
        rate_filter_min_timesteps=2,
        rate_filter_pad_timesteps=pad,
        rate_filter_plots=True,
        rate_filter_max_plots=2,
        chunk_size=100,
    )
    with caplog.at_level("INFO"):
        tractor_results = tukey_tractor(
            ms_path=Path(ms_example), tukey_tractor_options=tukey_tractor_options
        )

    # The example MS has few timesteps, so most segments can not be resolved
    # in delay-rate. Ensure segments were collected and considered.
    summaries = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(
            ("Delay-rate filtered", "Segments not filtered")
        )
    ]
    assert any(message.startswith("Delay-rate filtered") for message in summaries)
    assert any(message.startswith("Segments not filtered") for message in summaries)

    with table(str(ms_example), ack=False) as tab:
        assert new_column in tab.colnames()

    # Plots are only made of filtered segments, up to the maximum, at the end
    plots = sorted((Path(ms_example).parent / "plots").glob("*_rate_filter_*.png"))
    assert len(plots) <= 2
    assert tractor_results.rate_filter_plots is not None
    assert sorted(tractor_results.rate_filter_plots) == plots


def test_write_rate_filtered_segment(ms_example) -> None:
    """Only the nominated rows are written, including their restored flags"""
    rows = np.array([3, 7, 8])
    with table(str(ms_example), ack=False) as tab:
        data_shape = tab.getcell("DATA", 0).shape
        original_flags = tab.getcol("FLAG")
        original_weights = tab.getcol("WEIGHT")

    open_ms_tables = get_open_ms_tables(ms_path=Path(ms_example), read_only=False)
    tukey_tractor_options = TukeyTractorOptions(output_column="DATA")
    result = RateFilterResult(
        rows=rows,
        success=True,
        data=np.full((len(rows), *data_shape), 2 + 1j),
        flags=np.zeros((len(rows), *data_shape), dtype=bool),
        weights={"WEIGHT": np.full((len(rows), original_weights.shape[1]), 5.0)},
    )
    write_rate_filtered_segment(
        open_ms_tables=open_ms_tables,
        rate_filter_result=result,
        tukey_tractor_options=tukey_tractor_options,
    )
    # A failed segment is not written
    write_rate_filtered_segment(
        open_ms_tables=open_ms_tables,
        rate_filter_result=RateFilterResult(
            rows=np.array([0]), success=False, reason="too short"
        ),
        tukey_tractor_options=tukey_tractor_options,
    )
    open_ms_tables.close()

    with table(str(ms_example), ack=False) as tab:
        data = tab.getcol("DATA")
        flags = tab.getcol("FLAG")
        weights = tab.getcol("WEIGHT")

    assert np.all(data[rows] == 2 + 1j)
    others = np.setdiff1d(np.arange(len(data)), rows)
    assert not np.any(data[others] == 2 + 1j)
    assert np.all(weights[rows] == 5.0)
    np.testing.assert_array_equal(weights[others], original_weights[others])
    np.testing.assert_array_equal(flags[others], original_flags[others])
    # The example measurement set is entirely flagged, so the restored flags show
    assert np.all(original_flags[rows])
    assert not np.any(flags[rows])


def _segment_result(rows: list[int], value: float) -> RateFilterResult:
    n_rows = len(rows)
    return RateFilterResult(
        rows=np.array(rows),
        success=True,
        data=np.full((n_rows, 2, 2), value, dtype=complex),
        flags=np.full((n_rows, 2, 2), value > 1),
        weights={"WEIGHT": np.full((n_rows, 2), value)},
    )


def test_merge_rate_filter_results() -> None:
    merged = merge_rate_filter_results(
        [_segment_result([5, 9], 1.0), _segment_result([2, 7], 2.0)]
    )
    np.testing.assert_array_equal(merged.rows, [2, 5, 7, 9])
    assert merged.data is not None
    np.testing.assert_array_equal(merged.data[:, 0, 0], [2, 1, 2, 1])
    assert merged.flags is not None
    np.testing.assert_array_equal(merged.flags[:, 0, 0], [True, False, True, False])
    assert merged.weights is not None
    np.testing.assert_array_equal(merged.weights["WEIGHT"][:, 0], [2, 1, 2, 1])


def test_merge_rate_filter_results_without_weights() -> None:
    first, second = _segment_result([1], 1.0), _segment_result([0], 2.0)
    first.weights = second.weights = None
    assert merge_rate_filter_results([first, second]).weights is None


class _RecordingTable:
    """Stands in for the main table, recording selections that are written"""

    def __init__(self) -> None:
        self.writes: list[tuple[list[int], str]] = []

    def selectrows(self, rows):
        table = self

        class _Selection:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def putcol(self, column, _value):
                table.writes.append((list(rows), column))

        return _Selection()


def test_rate_filter_write_buffer() -> None:
    main_table = _RecordingTable()
    open_ms_tables = cast(Any, type("Tables", (), {"main_table": main_table})())
    buffer = RateFilterWriteBuffer(
        open_ms_tables=open_ms_tables,
        tukey_tractor_options=TukeyTractorOptions(output_column="OUT"),
        max_rows=4,
    )

    add_to_rate_filter_write_buffer(buffer, _segment_result([8, 9], 1.0))
    add_to_rate_filter_write_buffer(
        buffer, RateFilterResult(rows=np.array([0, 1, 2, 3]), success=False)
    )
    assert main_table.writes == []

    # Reaching max_rows writes all buffered segments as one sorted selection
    add_to_rate_filter_write_buffer(buffer, _segment_result([3, 4], 1.0))
    assert main_table.writes == [
        ([3, 4, 8, 9], "OUT"),
        ([3, 4, 8, 9], "FLAG"),
        ([3, 4, 8, 9], "WEIGHT"),
    ]

    add_to_rate_filter_write_buffer(buffer, _segment_result([6], 1.0))
    flush_rate_filter_write_buffer(buffer)
    assert main_table.writes[-3:] == [([6], "OUT"), ([6], "FLAG"), ([6], "WEIGHT")]

    # Nothing left to write
    n_writes = len(main_table.writes)
    flush_rate_filter_write_buffer(buffer)
    assert len(main_table.writes) == n_writes


@pytest.mark.parametrize("reweight", [False, True])
def test_tractor_reweight_opt_in(
    ms_example, monkeypatch: pytest.MonkeyPatch, reweight: bool
) -> None:
    """Weights are only modified when reweighting is requested"""
    with table(str(ms_example / "FIELD"), ack=False) as tab:
        phase_dir = tab.getcol("PHASE_DIR")[0, 0]
    phase = SkyCoord(*phase_dir, unit="rad")
    target = SkyCoord(phase.ra, phase.dec + 8 * u.deg)
    monkeypatch.setattr(SkyCoord, "from_name", staticmethod(lambda _: target))

    with table(str(ms_example), ack=False) as tab:
        original_weights = tab.getcol("WEIGHT")

    tukey_tractor(
        ms_path=Path(ms_example),
        tukey_tractor_options=TukeyTractorOptions(
            target_objects=("NEAR_FIELD",),
            outer_width_ns=4.0,
            tukey_width_ns=2.0,
            output_column="JACKS_DATA",
            reweight=reweight,
        ),
    )

    with table(str(ms_example), ack=False) as tab:
        weights = tab.getcol("WEIGHT")

    if reweight:
        assert not np.array_equal(weights, original_weights)
    else:
        np.testing.assert_array_equal(weights, original_weights)


def test_compute_tukey_multi_taper_later_object_below_elevation() -> None:
    """An object that returns early must not discard the delay-time formed by an
    earlier object, which is needed to apply the taper"""
    n_time = 8
    above = _make_w_delays(n_time=n_time)
    below = _make_w_delays(n_time=n_time, elevation_deg=-10.0)

    for w_delays_list in ([above, below], [below, above]):
        result = compute_tukey_multi_taper(
            data_chunk=_make_data_chunk(n_time=n_time),
            tukey_tractor_options=TukeyTractorOptions(),
            w_delays_list=w_delays_list,
        )
        assert not result.nothing_to_do
        assert result.update_data


def test_compute_auto_taper_widths_overrides_rate_width() -> None:
    """As for the delay widths, auto sizing overrides a requested delay-rate width"""
    options = compute_auto_taper_widths(
        freq_chan=np.linspace(0.8, 1.1, 64) * u.GHz,
        tukey_tractor_options=TukeyTractorOptions(
            auto_size=True, rate_filter=True, rate_filter_width_hz=0.01
        ),
    )
    assert options.rate_filter_width_hz is None
    assert options.tukey_width_ns == 0.0


@pytest.mark.parametrize(
    ("auto_size", "nth_sidelobe_null", "expected_sidelobes"),
    [(False, None, 1), (True, None, 1), (True, 3, 3)],
)
def test_rate_filter_settings_auto(
    auto_size: bool, nth_sidelobe_null: int | None, expected_sidelobes: int
) -> None:
    settings = _rate_filter_settings(
        TukeyTractorOptions(auto_size=auto_size, nth_sidelobe_null=nth_sidelobe_null)
    )
    assert settings.auto_width is auto_size
    assert settings.auto_sidelobes == expected_sidelobes


def _rate_filter_processor_with_segments(
    tmp_path: Path, n_baselines: int, max_plots: int
) -> RateFilterProcessor:
    """A processor whose accumulator holds one filterable segment per baseline. The
    object crosses delay zero at a constant delay-rate, separable from the field."""
    n_time, n_chan, dt_s = 64, 64, 10.0
    freq_hz = np.linspace(0.8e9, 1.1e9, n_chan)
    time_s = 5e9 + np.arange(n_time) * dt_s
    tau_s = 2e-11 * (time_s - time_s.mean())
    vis = 1.0 + 5.0 * np.exp(2j * np.pi * freq_hz[None, :] * tau_s[:, None])
    vis = np.repeat(vis[..., None], 2, axis=-1)

    w_delays = WDelays(
        object_name="sun",
        w_delays=np.repeat((tau_s * u.s)[None, :], n_baselines, axis=0),
        b_map={(0, b + 1): b for b in range(n_baselines)},
        time_map={t * u.s: idx for idx, t in enumerate(time_s)},
        elevation=np.full(n_time, 45.0) * u.deg,
    )
    options = TukeyTractorOptions(
        outer_width_ns=10.0,
        tukey_width_ns=5.0,
        rate_filter=True,
        rate_filter_plots=True,
        rate_filter_max_plots=max_plots,
    )
    open_ms_tables = cast(
        Any,
        type(
            "Tables",
            (),
            {"main_table": _RecordingTable(), "ms_path": tmp_path / "test.ms"},
        )(),
    )
    processor = RateFilterProcessor(
        open_ms_tables=open_ms_tables,
        tukey_tractor_options=options,
        w_delays_list=[w_delays],
        settings=_rate_filter_settings(options),
        freq_chan=freq_hz * u.Hz,
        accumulator=SegmentAccumulator(),
        write_buffer=RateFilterWriteBuffer(
            open_ms_tables=open_ms_tables,
            tukey_tractor_options=options,
            max_rows=10_000,
        ),
    )
    for baseline in range(n_baselines):
        update_segment_accumulator(
            accumulator=processor.accumulator,
            row_numbers=baseline * n_time + np.arange(n_time),
            ant_1=np.zeros(n_time, dtype=int),
            ant_2=np.full(n_time, baseline + 1),
            baseline_idx=np.full(n_time, baseline),
            time_mjds=time_s,
            time_idx=np.arange(n_time),
            data=vis,
            mask=np.zeros(vis.shape, dtype=bool),
            recoverable=np.ones(n_time, dtype=bool),
            delay_contaminated=np.ones(n_time, dtype=bool),
        )
    return processor


def test_rate_filter_plots_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Segments are not plotted while processing, only once it has finished, and
    no more than the maximum number of plots are made"""
    plotted: list[Path] = []

    def _record(output_path: Path, **_: Any) -> Path:
        plotted.append(output_path)
        return output_path

    monkeypatch.setattr("jolly_roger.tractor.plot_rate_filter_segment", _record)
    processor = _rate_filter_processor_with_segments(
        tmp_path=tmp_path, n_baselines=3, max_plots=2
    )

    finish_rate_filter(processor)
    assert processor.summary.segments_filtered == 3
    assert plotted == []
    assert len(processor.diagnostics) == 2

    plot_paths = make_rate_filter_plots(processor)
    assert plot_paths == plotted
    assert len(plot_paths) == 2
    assert all(path.parent == tmp_path / "plots" for path in plot_paths)
    # The kept diagnostics are released once plotted
    assert processor.diagnostics == []


@pytest.mark.parametrize("rate_filter", [False, True])
def test_tractor_make_plots_delay_rate(ms_example, rate_filter: bool) -> None:
    """With delay-rate filtering each plotted baseline also has a delay-rate figure"""
    tractor_results = tukey_tractor(
        ms_path=Path(ms_example),
        tukey_tractor_options=TukeyTractorOptions(
            outer_width_ns=4.0,
            tukey_width_ns=2.0,
            output_column="JACKS_DATA",
            make_plots=True,
            number_of_plots=2,
            rate_filter=rate_filter,
        ),
    )

    assert tractor_results.output_plots is not None
    names = [path.name for path in tractor_results.output_plots]
    assert all(path.exists() for path in tractor_results.output_plots)
    comparison = [name for name in names if name.endswith("_comparison.png")]
    delay_rate = [name for name in names if name.endswith("_delay_rate_comparison.png")]
    assert len(comparison) - len(delay_rate) == 2
    assert len(delay_rate) == (2 if rate_filter else 0)


def test_rate_filter_processor_records_applied_notches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regions nulled are kept per baseline for the delay-rate comparison"""
    monkeypatch.setattr(
        "jolly_roger.tractor.plot_rate_filter_segment",
        lambda output_path, **_: output_path,
    )
    processor = _rate_filter_processor_with_segments(
        tmp_path=tmp_path, n_baselines=3, max_plots=1
    )
    finish_rate_filter(processor)

    assert sorted(processor.applied_notches) == [(0, 1), (0, 2), (0, 3)]
    for notches in processor.applied_notches.values():
        (footprint,) = notches
        assert footprint.object_name == "sun"


@pytest.mark.parametrize("limit", [2, None])
def test_rate_filter_settings_rate_nyquist_zone(limit: int | None) -> None:
    assert TukeyTractorOptions().rate_filter_ignore_nyquist_zone == 2
    settings = _rate_filter_settings(
        TukeyTractorOptions(rate_filter_ignore_nyquist_zone=limit)
    )
    assert settings.ignore_rate_nyquist_zone == limit
