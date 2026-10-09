"""Tests around the diagnostic figures"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib.patches import Rectangle

from jolly_roger.baselines import BaselineData
from jolly_roger.delays import data_to_delay_rate, data_to_delay_time
from jolly_roger.plots import (
    _taper_extent_mask,
    plot_baseline_comparison_data,
    plot_baseline_delay_rate_comparison,
)
from jolly_roger.rates import RateBox
from jolly_roger.uvws import WDelays

N_TIME = 32
N_CHAN = 64


def _baseline_inputs() -> tuple[BaselineData, BaselineData, WDelays]:
    """A field source and an object whose delay crosses zero at a constant rate"""
    rng = np.random.default_rng(1934)
    freq_hz = np.linspace(0.8e9, 1.1e9, N_CHAN)
    time_s = 5.2e9 + np.arange(N_TIME) * 10.0
    tau_s = 2e-11 * (time_s - time_s.mean())
    vis = (
        1.0
        + 5.0 * np.exp(2j * np.pi * freq_hz[None, :] * tau_s[:, None])
        + 0.1 * rng.standard_normal((N_TIME, N_CHAN))
    )
    vis = np.repeat(vis[..., None], 4, axis=-1)

    def _baseline_data(data: np.ndarray) -> BaselineData:
        return BaselineData(
            masked_data=np.ma.masked_array(data, mask=np.zeros(data.shape, bool)),
            freq_chan=freq_hz * u.Hz,
            phase_center=SkyCoord(0 * u.deg, 0 * u.deg),
            uvws_phase_center=np.zeros((3, N_TIME)) * u.m,
            time=Time(time_s / 86400.0, format="mjd", scale="utc"),
            ant_1=0,
            ant_2=1,
        )

    w_delays = WDelays(
        object_name="sun",
        w_delays=(tau_s * u.s)[None, :],
        b_map={(0, 1): 0},
        time_map={t * u.s: idx for idx, t in enumerate(time_s)},
        elevation=np.linspace(10.0, 30.0, N_TIME) * u.deg,
        guard_region=np.full((1, N_TIME), 3e-9) * u.s,
        w_rates=np.full((1, N_TIME), 2e-11) * u.dimensionless_unscaled,
        rate_guard_region=np.full((1, N_TIME), 3e-12) * u.dimensionless_unscaled,
    )
    return _baseline_data(vis), _baseline_data(vis * 0.5), w_delays


def test_plot_baseline_comparison_data(tmp_path: Path) -> None:
    before, after, w_delays = _baseline_inputs()
    output_path = plot_baseline_comparison_data(
        before_baseline_data=before,
        after_baseline_data=after,
        before_delays=data_to_delay_time(before),
        after_delays=data_to_delay_time(after),
        output_path=tmp_path / "comparison.png",
        w_delays=w_delays,
        outer_width_ns=10.0,
    )
    assert output_path.exists()
    plt.close("all")


@pytest.mark.parametrize("with_objects", [True, False])
def test_plot_baseline_delay_rate_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_objects: bool
) -> None:
    """The bottom row spans delay and delay-rate, while the top row is in time"""
    before, after, w_delays = _baseline_inputs()
    before_delay_rate = data_to_delay_rate(before)

    figures: list[plt.Figure] = []
    monkeypatch.setattr("jolly_roger.plots.plt.close", figures.append)

    output_path = plot_baseline_delay_rate_comparison(
        before_baseline_data=before,
        after_baseline_data=after,
        before_delay_rate=before_delay_rate,
        after_delay_rate=data_to_delay_rate(after),
        output_path=tmp_path / "delay_rate_comparison.png",
        w_delays=w_delays if with_objects else None,
        outer_width_ns=10.0,
    )
    assert output_path.exists()

    (figure,) = figures
    axes = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
    dynamic_spectra = [ax for ax in axes if ax.get_ylabel().startswith("Frequency")]
    bottom = [ax for ax in axes if ax.get_ylabel() == "Delay / ns"]
    assert len(dynamic_spectra) == 2
    assert len(bottom) == 3
    # The dynamic spectra remain in time
    assert dynamic_spectra[0].get_shared_x_axes().joined(*dynamic_spectra)
    assert (
        not dynamic_spectra[0].get_shared_x_axes().joined(dynamic_spectra[0], bottom[0])
    )

    delay_ns = before_delay_rate.delay.to("ns").value
    rate_mhz = before_delay_rate.rate.to("mHz").value
    for ax in bottom:
        # Delay is along the y-axis, as in the delay vs time panels
        assert ax.get_xlabel() == "Fringe-rate / mHz"
        np.testing.assert_allclose(
            sorted(ax.get_xlim()), [rate_mhz.min(), rate_mhz.max()], rtol=0.05
        )
        np.testing.assert_allclose(
            sorted(ax.get_ylim()), [delay_ns.min(), delay_ns.max()], rtol=0.05
        )

    track_panel = bottom[1]
    labels = track_panel.get_legend_handles_labels()[1]
    assert "Field" in labels
    assert "Guard Region" in labels
    assert ("Path of sun" in labels) is with_objects
    plt.close(figure)


def test_taper_extent_mask() -> None:
    """Cells within the delay width and the fringe-rate spread of the object, at
    any time, are covered. Narrow extents still cover the cell they pass through."""
    delay_ns = np.arange(-50.0, 50.0, 1.0)
    rate_mhz = np.arange(-20.0, 20.0, 1.0)
    mask = _taper_extent_mask(
        delay_ns=delay_ns,
        rate_mhz=rate_mhz,
        object_delay_ns=np.array([-5.0, 5.0]),
        object_rate_low_mhz=np.array([8.0, 8.0]),
        object_rate_high_mhz=np.array([12.0, 12.0]),
        width_ns=3.0,
    )
    assert mask.shape == (len(delay_ns), len(rate_mhz))

    covered_delay = delay_ns[np.any(mask, axis=1)]
    covered_rate = rate_mhz[np.any(mask, axis=0)]
    np.testing.assert_array_equal(
        covered_delay, [-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8]
    )
    np.testing.assert_array_equal(covered_rate, [8, 9, 10, 11, 12])

    narrow = _taper_extent_mask(
        delay_ns=delay_ns,
        rate_mhz=rate_mhz,
        object_delay_ns=np.array([0.2]),
        object_rate_low_mhz=np.array([10.2]),
        object_rate_high_mhz=np.array([10.2]),
        width_ns=0.0,
    )
    assert narrow.sum() == 1
    assert narrow[delay_ns == 0.0, rate_mhz == 10.0].all()


def _object_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    applied_notches: list[tuple[str, RateBox]] | None,
) -> plt.Axes:
    """Render the delay-rate comparison and return its object panel"""
    before, after, w_delays = _baseline_inputs()
    figures: list[plt.Figure] = []
    monkeypatch.setattr("jolly_roger.plots.plt.close", figures.append)
    plot_baseline_delay_rate_comparison(
        before_baseline_data=before,
        after_baseline_data=after,
        before_delay_rate=data_to_delay_rate(before),
        after_delay_rate=data_to_delay_rate(after),
        output_path=tmp_path / "delay_rate_comparison.png",
        w_delays=w_delays,
        outer_width_ns=10.0,
        applied_notches=applied_notches,
    )
    (figure,) = figures
    return next(
        ax
        for ax in figure.axes
        if ax.get_ylabel() == "Delay / ns" and not ax.get_title()
    )


def _dashed_notches(ax: plt.Axes) -> list[Rectangle]:
    return [
        patch
        for patch in ax.patches
        if isinstance(patch, Rectangle)
        and patch.get_linestyle() == "--"
        and not patch.get_fill()
    ]


def test_delay_rate_comparison_shows_taper_extent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ax = _object_panel(tmp_path, monkeypatch, applied_notches=None)
    labels = ax.get_legend_handles_labels()[1]
    assert "sun taper extent" in labels
    assert "Applied notch" not in labels
    assert _dashed_notches(ax) == []
    # The filled extent and its outline
    assert len(ax.collections) >= 2
    plt.close("all")


def test_delay_rate_comparison_shows_applied_notches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notches = [
        ("sun", RateBox(-5e-9, 10e-9, 0.019, 0.004)),
        ("sun", RateBox(5e-9, 10e-9, 0.019, 0.004)),
    ]
    ax = _object_panel(tmp_path, monkeypatch, applied_notches=notches)
    labels = ax.get_legend_handles_labels()[1]
    assert labels.count("Applied notch") == 1

    drawn = _dashed_notches(ax)
    assert len(drawn) == 2
    # Fringe-rate (mHz) along x and delay (ns) along y
    first = drawn[0]
    assert first.get_x() == pytest.approx((0.019 - 0.004) * 1e3)
    assert first.get_y() == pytest.approx((-5e-9 - 10e-9) * 1e9)
    assert first.get_width() == pytest.approx(2 * 0.004 * 1e3)
    assert first.get_height() == pytest.approx(2 * 10e-9 * 1e9)
    plt.close("all")


def test_delay_rate_comparison_path_wraps_in_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An object whose fringe-rate sweeps beyond the edge of the panel is drawn
    for the whole observation, aliased back into the panel as the data are,
    rather than leaving the panel"""
    before, after, w_delays = _baseline_inputs()
    # 0.95 GHz x 2e-11 ~ 19 mHz up to x 2e-10 ~ 190 mHz, beyond the 50 mHz edge
    sweeping = replace(
        w_delays,
        w_rates=np.linspace(2e-11, 2e-10, N_TIME)[None, :] * u.dimensionless_unscaled,
    )
    before_delay_rate = data_to_delay_rate(before)
    rate_mhz = before_delay_rate.rate.to("mHz").value

    figures: list[plt.Figure] = []
    monkeypatch.setattr("jolly_roger.plots.plt.close", figures.append)
    plot_baseline_delay_rate_comparison(
        before_baseline_data=before,
        after_baseline_data=after,
        before_delay_rate=before_delay_rate,
        after_delay_rate=data_to_delay_rate(after),
        output_path=tmp_path / "delay_rate_comparison.png",
        w_delays=sweeping,
        outer_width_ns=10.0,
    )
    (figure,) = figures
    ax = next(
        ax
        for ax in figure.axes
        if ax.get_ylabel() == "Delay / ns" and not ax.get_title()
    )
    path_segments = [line for line in ax.lines if line.get_linewidth() == 3]

    # Every timestep is drawn once, within the panel, in several wrapped pieces
    assert sum(len(line.get_xdata()) for line in path_segments) == N_TIME
    assert len(path_segments) > 1
    for line in path_segments:
        x = np.asarray(line.get_xdata())
        assert np.all(x >= rate_mhz.min() - 1e-9)
        assert np.all(x <= rate_mhz.max() + 1e-9)
    plt.close(figure)
