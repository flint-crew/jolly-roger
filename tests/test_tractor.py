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
    times = (np.arange(200) - 100) * u.s
    width_ns = 10

    mask = make_search_window(x=times, width_ns=width_ns)
    assert np.sum(mask) == 19
    assert np.all(~mask[:91])
    assert np.all(mask[91 : 91 + 19])
    assert np.all(~mask[91 + 19 :])


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
