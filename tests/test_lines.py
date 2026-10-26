"""Tests for the spectral-line impact estimates"""

from __future__ import annotations

import numpy as np
import pytest

from jolly_roger.lines import (
    describe_line_impact,
    gaussian_line_peak_loss,
    summarise_line_impact,
)


def test_gaussian_line_peak_loss_matches_notch() -> None:
    """The analytic peak loss agrees with notching a sampled Gaussian line"""
    n_chan, chan_width_hz = 8192, 25e3
    freq_hz = (np.arange(n_chan) - n_chan // 2) * chan_width_hz
    fwhm_hz, effective_width_s, object_s = 1e6, 20e-9, 0.4e-6
    line = np.exp(-4 * np.log(2) * (freq_hz / fwhm_hz) ** 2)

    delay_s = np.fft.fftfreq(n_chan, d=chan_width_hz)
    # A box notch of the same effective width, centred on the object
    removed = np.abs(delay_s - object_s) < effective_width_s / 2
    filtered = np.fft.ifft(np.fft.fft(line) * ~removed).real

    measured = 1.0 - filtered[n_chan // 2]
    expected = gaussian_line_peak_loss(effective_width_s, fwhm_hz, object_s)
    assert measured == pytest.approx(float(expected), rel=0.05)


def test_summarise_line_impact_worst_case() -> None:
    """The worst case over line widths matches a brute-force search"""
    effective_width_s, separation_s = 10e-9, 300e-9
    summary = summarise_line_impact(
        delay_separation_s=np.array([separation_s, 2e-6, -5e-6]),
        effective_width_s=effective_width_s,
        channel_width_hz=18.5e3,
        reference_frequency_hz=1.4e9,
    )

    widths_hz = np.linspace(1e3, 20e6, 200_000)
    losses = gaussian_line_peak_loss(effective_width_s, widths_hz, separation_s)
    assert summary.worst_loss == pytest.approx(losses.max(), rel=1e-3)
    assert summary.worst_line_fwhm_hz == pytest.approx(
        widths_hz[np.argmax(losses)], rel=1e-3
    )
    assert summary.closest_delay_s == pytest.approx(separation_s)
    assert summary.unresolved_loss == pytest.approx(effective_width_s * 18.5e3)
    assert summary.number_of_visibilities == 3


def test_summarise_line_impact_fraction_and_empty() -> None:
    """Visibilities closer than about 24 W in delay can lose more than 1%"""
    effective_width_s = 10e-9
    summary = summarise_line_impact(
        delay_separation_s=np.array([100e-9, 200e-9, 300e-9, 1e-6]),
        effective_width_s=effective_width_s,
        channel_width_hz=1e6,
        reference_frequency_hz=1e9,
    )
    assert summary.fraction_above_one_percent == pytest.approx(0.5)

    empty = summarise_line_impact(
        delay_separation_s=np.array([]),
        effective_width_s=effective_width_s,
        channel_width_hz=1e6,
        reference_frequency_hz=1e9,
    )
    assert empty.number_of_visibilities == 0
    assert "No visibilities were modified" in describe_line_impact("sun", empty)


def test_describe_line_impact() -> None:
    """The description names the object and quotes widths in frequency and velocity"""
    summary = summarise_line_impact(
        delay_separation_s=np.array([77e-9]),
        effective_width_s=10e-9,
        channel_width_hz=18.5e3,
        reference_frequency_hz=1.4e9,
    )
    text = describe_line_impact("sun", summary)
    assert text.startswith("Spectral-line impact of nulling sun")
    assert "77 ns" in text
    assert "km/s at 1.4 GHz" in text
    assert "Integrated line flux is unchanged." in text
