"""Tests around the tractor'ing. Some are simple, some are complex, but
all are important in their own way"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from casacore.tables import table
from numpy import ma
from numpy.typing import NDArray

from jolly_roger.delays import data_to_delay_time
from jolly_roger.tractor import (
    DataChunk,
    TukeyTractorOptions,
    apply_roll_for_taper,
    compute_tukey_multi_taper,
    find_idx_of_closest_delay,
    make_search_window,
    tukey_tractor,
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
        time=Time.now(),
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


def _peak_search_chunk(
    source_ns: float,
    source_amp: float,
    field_amp: float = 1.0,
    n_time: int = 4,
    n_chan: int = 256,
    n_pol: int = 4,
) -> tuple[DataChunk, NDArray[np.floating]]:
    """A single-baseline chunk with a field peak at delay 0 and an optional
    source peak at ``source_ns``, built directly in delay space (matching
    ``data_to_delay_time``'s forward transform)."""
    freq_chan = np.linspace(744, 1032, n_chan) * u.MHz
    delay_ns = (
        np.fft.fftshift(np.fft.fftfreq(n_chan, d=np.diff(freq_chan).mean()).decompose())
        .to(u.ns)
        .value
    )
    delay_space = np.zeros((n_time, n_chan), dtype=complex)
    delay_space[:, np.abs(delay_ns).argmin()] += field_amp
    if source_amp:
        delay_space[:, np.abs(delay_ns - source_ns).argmin()] += source_amp
    vis = np.fft.ifft(np.fft.ifftshift(delay_space, axes=1), axis=1, norm="forward")

    chunk = DataChunk(
        masked_data=ma.masked_array(
            vis[..., None].repeat(n_pol, -1),
            mask=np.zeros((n_time, n_chan, n_pol), dtype=bool),
        ),
        freq_chan=freq_chan,
        phase_center=SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg),
        uvws_phase_center=np.zeros((n_time, 3)) * u.m,
        time=Time.now(),
        time_mjds=np.arange(n_time, dtype=float),
        ant_1=np.zeros(n_time, dtype=np.int64),
        ant_2=np.ones(n_time, dtype=np.int64),
        row_start=0,
        chunk_size=n_time,
    )
    return chunk, delay_ns


def _predict_field_wdelays(n_time: int, guard_ns: float | None = None) -> WDelays:
    """WDelays predicting the object at delay 0 (the field), optionally with a
    protected guard band of half-width ``guard_ns``."""
    guard = None if guard_ns is None else np.full((1, n_time), guard_ns * 1e-9) * u.s
    return WDelays(
        object_name="drifter",
        w_delays=np.zeros((1, n_time)) * u.s,
        b_map={(0, 1): 0},
        time_map={t * u.s: idx for idx, t in enumerate(np.arange(n_time, dtype=float))},
        elevation=np.full(n_time, 90.0) * u.deg,
        guard_region=guard,
    )


def _delay_amp(
    chunk: DataChunk | None, delay_ns: NDArray[np.floating], target_ns: float
) -> float:
    """Peak |delay spectrum| (pol 0) in a small window around ``target_ns``."""
    assert chunk is not None
    delay_time = data_to_delay_time(chunk)
    bin_idx = int(np.abs(delay_ns - target_ns).argmin())
    window = slice(bin_idx - 2, bin_idx + 3)
    return float(np.abs(delay_time.delay_time[:, window, 0]).max())


def test_peak_shift_search_tracks_bright_source() -> None:
    """A source that out-shines the field is found in the search window and the
    null is moved onto it, sparing the field at delay 0. No detection opt-in is
    required for a genuine shift."""
    n_time, source_ns = 4, 80.0
    options = TukeyTractorOptions(
        outer_width_ns=40.0,
        tukey_width_ns=10.0,
        peak_shift_search=True,
        peak_shift_search_width_ns=120.0,
    )

    chunk, delay_ns = _peak_search_chunk(source_ns=source_ns, source_amp=10.0)
    result = compute_tukey_multi_taper(chunk, options, [_predict_field_wdelays(n_time)])
    assert _delay_amp(result.data_chunk, delay_ns, source_ns) < 0.1
    assert _delay_amp(result.data_chunk, delay_ns, 0.0) > 0.5


def test_peak_shift_search_flags_object_over_field() -> None:
    """When the object is predicted onto the field at delay 0 the crossing must
    be flagged, and the field data must be preserved (never nulled in the guard
    region). This is the default path with no detection opt-in."""
    n_time = 4
    options = TukeyTractorOptions(
        outer_width_ns=40.0,
        tukey_width_ns=10.0,
        peak_shift_search=True,
        peak_shift_search_width_ns=120.0,
    )

    chunk, delay_ns = _peak_search_chunk(source_ns=0.0, source_amp=0.0)
    result = compute_tukey_multi_taper(chunk, options, [_predict_field_wdelays(n_time)])
    assert result.flags is not None
    assert result.flags[:, :, 0].all()
    assert _delay_amp(result.data_chunk, delay_ns, 0.0) > 0.5


def test_peak_shift_search_compare_to_field_ignores_faint() -> None:
    """With compare_to_field opted in, a source fainter than the field is not a
    detection, so the shift falls back to the predicted position over the field:
    the crossing is flagged, the field is preserved, and the faint source is
    left alone."""
    n_time, source_ns = 4, 80.0
    options = TukeyTractorOptions(
        outer_width_ns=40.0,
        tukey_width_ns=10.0,
        peak_shift_search=True,
        peak_shift_search_width_ns=120.0,
        compare_to_field=0.2,
    )

    chunk, delay_ns = _peak_search_chunk(source_ns=source_ns, source_amp=0.1)
    result = compute_tukey_multi_taper(chunk, options, [_predict_field_wdelays(n_time)])
    assert result.flags is not None
    assert result.flags[:, :, 0].all()
    assert _delay_amp(result.data_chunk, delay_ns, source_ns) > 0.05
    assert _delay_amp(result.data_chunk, delay_ns, 0.0) > 0.5


def test_peak_shift_search_never_nulls_guard_band() -> None:
    """The object null is never applied inside the guard region. A bright source
    outside the field null is nulled normally, but once a guard band covers it
    the null is clamped away and the source survives (the crossing is flagged
    instead)."""
    n_time, source_ns = 4, 60.0
    options = TukeyTractorOptions(
        outer_width_ns=40.0,
        tukey_width_ns=10.0,
        peak_shift_search=True,
        peak_shift_search_width_ns=120.0,
    )

    # No guard: the source sits outside the field null, is found, and is nulled.
    chunk, delay_ns = _peak_search_chunk(source_ns=source_ns, source_amp=10.0)
    result = compute_tukey_multi_taper(chunk, options, [_predict_field_wdelays(n_time)])
    assert _delay_amp(result.data_chunk, delay_ns, source_ns) < 0.1

    # Guard band covering the source: the null is clamped out of the guard, so
    # the source survives and the row is flagged instead.
    chunk, delay_ns = _peak_search_chunk(source_ns=source_ns, source_amp=10.0)
    result = compute_tukey_multi_taper(
        chunk, options, [_predict_field_wdelays(n_time, guard_ns=60.0)]
    )
    assert result.flags is not None
    assert result.flags[:, :, 0].all()
    assert _delay_amp(result.data_chunk, delay_ns, source_ns) > 5.0


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
