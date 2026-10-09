"""Tests around the diagnostic figures"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jolly_roger.baselines import BaselineData
from jolly_roger.delays import data_to_delay_rate, data_to_delay_time
from jolly_roger.plots import (
    plot_baseline_comparison_data,
    plot_baseline_delay_rate_comparison,
)
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
