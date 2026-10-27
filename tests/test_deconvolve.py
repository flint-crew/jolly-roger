"""Tests for removing objects outside the field by CLEAN in delay"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from astropy import units as u
from numpy import ma
from numpy.typing import NDArray

from jolly_roger.deconvolve import (
    DelayCleanOptions,
    MissingCleanDependencyError,
    compute_delay_clean,
    delay_clean_spectrum,
    estimate_visibility_noise,
)
from jolly_roger.tractor import DataChunk

N_CHAN = 128
CHANNEL_WIDTH_HZ = 2e6
FREQ_HZ = 1.2e9 + np.arange(N_CHAN) * CHANNEL_WIDTH_HZ
FIELD_DELAY_S = 30e-9
SIGMA = 0.01


def _fringe(delay_s: float) -> NDArray[np.complexfloating]:
    return np.exp(2j * np.pi * FREQ_HZ * delay_s)


def _field() -> NDArray[np.complexfloating]:
    """Sources near the phase centre, with a spectral index"""
    spectrum = (FREQ_HZ / FREQ_HZ.mean()) ** -0.7
    return spectrum * (
        1.0 * _fringe(0.0) + 0.5 * _fringe(10e-9) + 0.3 * _fringe(-15e-9)
    )


def _resolved_object(
    delay_s: float, amplitude: float = 10.0
) -> NDArray[np.complexfloating]:
    """An extended object (40 ns across) with a steep, rippled spectrum. Its
    size and spectrum are not given to the CLEAN."""
    offsets = np.linspace(-20e-9, 20e-9, 9)
    profile = np.hanning(11)[1:-1]
    values = sum(
        p * _fringe(delay_s + o) for p, o in zip(profile, offsets, strict=True)
    )
    values *= (FREQ_HZ / FREQ_HZ.mean()) ** -2.5
    values *= 1 + 0.3 * np.cos(2 * np.pi * FREQ_HZ * 25e-9)
    return amplitude * values / np.sqrt(np.mean(np.abs(values) ** 2))


def _noise(seed: int) -> NDArray[np.complexfloating]:
    rng = np.random.default_rng(seed)
    return SIGMA * (rng.standard_normal(N_CHAN) + 1j * rng.standard_normal(N_CHAN))


def _rms(values: NDArray[np.complexfloating]) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


# Median amplitude of complex noise with SIGMA per real or imaginary part
NOISE_MEDIAN = SIGMA * np.sqrt(2 * np.log(2))


def test_resolved_object_is_removed_and_field_kept() -> None:
    truth = _field() + _noise(1)
    weights = np.ones(N_CHAN)
    weights[40:46] = 0.0

    result = delay_clean_spectrum(
        values=truth + _resolved_object(150e-9),
        weights=weights,
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=FIELD_DELAY_S,
        visibility_noise=SIGMA,
    )
    cleaned = truth + _resolved_object(150e-9) - result.model

    good = weights > 0
    noise_rms = SIGMA * np.sqrt(2)
    # The object (rms 10) is removed to about the noise in most channels. What
    # is left is largest in the few channels at the band edges, where the
    # object is cut off and CLEAN stops at its threshold.
    residual = np.abs((cleaned - truth)[good])
    assert np.median(residual) < 2 * NOISE_MEDIAN
    assert _rms(residual) < 5 * noise_rms
    assert not result.reaches_field

    # With no object present, nothing in the field is subtracted
    field_only = delay_clean_spectrum(
        values=truth,
        weights=weights,
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=FIELD_DELAY_S,
        visibility_noise=SIGMA,
    )
    assert _rms(field_only.model) < 0.5 * noise_rms


def test_object_is_removed_beneath_flagged_channels() -> None:
    """Flagged channels do not enter the CLEAN, but the model is still evaluated there"""
    weights = np.ones(N_CHAN)
    weights[40:46] = 0.0
    values = _field() + _resolved_object(150e-9)
    corrupted = values.copy()
    corrupted[40:46] = 1e6

    result = delay_clean_spectrum(
        values=corrupted,
        weights=weights,
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=FIELD_DELAY_S,
        visibility_noise=SIGMA,
    )

    # The object (rms 10) is interpolated across the gap to within a few per cent
    assert _rms((values - result.model - _field())[40:46]) < 0.5


def test_noise_estimate_of_pure_noise() -> None:
    rng = np.random.default_rng(5)
    values = SIGMA * (rng.standard_normal(1024) + 1j * rng.standard_normal(1024))

    estimate = estimate_visibility_noise(values, np.ones(1024))

    assert estimate is not None
    assert 0.9 * SIGMA < estimate < 1.1 * SIGMA


def test_noise_estimate_needs_long_enough_runs_of_channels() -> None:
    weights = np.ones(N_CHAN)
    weights[::20] = 0.0

    assert estimate_visibility_noise(_noise(6), weights) is None


def test_noise_is_estimated_when_not_given() -> None:
    """A bright object (10^3 times the noise) raises the estimate only a little"""
    truth = _field() + _noise(2)
    weights = np.ones(N_CHAN)
    weights[40:46] = 0.0
    values = truth + _resolved_object(-180e-9)

    estimate = estimate_visibility_noise(values, weights)
    assert estimate is not None
    assert 0.8 * SIGMA < estimate < 2.0 * SIGMA

    result = delay_clean_spectrum(
        values=values,
        weights=weights,
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=FIELD_DELAY_S,
    )
    residual = np.abs((values - result.model - truth)[weights > 0])
    assert np.median(residual) < 3 * NOISE_MEDIAN


def test_noise_only_subtracts_nothing() -> None:
    result = delay_clean_spectrum(
        values=_noise(3),
        weights=np.ones(N_CHAN),
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=FIELD_DELAY_S,
        visibility_noise=SIGMA,
    )

    assert _rms(result.model) < 0.2 * SIGMA
    assert not result.reaches_field


def test_object_next_to_field_is_reported() -> None:
    result = delay_clean_spectrum(
        values=_field() + _noise(4) + _resolved_object(55e-9),
        weights=np.ones(N_CHAN),
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=FIELD_DELAY_S,
        visibility_noise=SIGMA,
    )

    assert result.reaches_field


def test_fully_flagged_spectrum_is_left_alone() -> None:
    result = delay_clean_spectrum(
        values=_field(),
        weights=np.zeros(N_CHAN),
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=FIELD_DELAY_S,
    )

    assert result.iterations == 0
    assert np.all(result.model == 0)


def _chunk(object_delays_s: list[float]) -> DataChunk:
    n_rows, n_pol = len(object_delays_s), 2
    data = np.zeros((n_rows, N_CHAN, n_pol), dtype=complex)
    for row, delay_s in enumerate(object_delays_s):
        for pol in range(n_pol):
            data[row, :, pol] = (
                _field() + _noise(10 * row + pol) + _resolved_object(delay_s)
            )
    mask = np.zeros(data.shape, dtype=bool)
    mask[:, 40:46, :] = True
    return DataChunk(
        masked_data=ma.masked_array(data, mask=mask),
        freq_chan=FREQ_HZ * u.Hz,
        time_mjds=np.arange(n_rows, dtype=float),
        ant_1=np.zeros(n_rows, dtype=np.int64),
        ant_2=np.ones(n_rows, dtype=np.int64),
        chunk_size=n_rows,
    )


def test_compute_delay_clean_on_chunk() -> None:
    data_chunk = _chunk([150e-9, 55e-9])

    result = compute_delay_clean(
        data_chunk=data_chunk,
        field_delay_s=FIELD_DELAY_S,
        options=DelayCleanOptions(),
        visibility_noise=SIGMA,
    )

    assert result.update_data
    assert result.data_chunk is not None
    cleaned = result.data_chunk.masked_data
    for pol in range(2):
        expected = _field() + _noise(pol)
        good = ~cleaned.mask[0, :, pol]
        residual = np.abs((cleaned.data[0, :, pol] - expected)[good])
        assert np.median(residual) < 2 * NOISE_MEDIAN

    # The object next to the field flags its whole row, and only that row
    assert result.update_flags
    assert result.flags is not None
    assert result.flags[1].all()
    assert not result.flags[0, :40].any()
    # Existing flags are kept
    assert result.flags[0, 40:46].all()


def test_missing_rm_lite_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rm_lite.utils", None)

    with pytest.raises(MissingCleanDependencyError, match=r"jolly-roger\[clean\]"):
        delay_clean_spectrum(
            values=_field(),
            weights=np.ones(N_CHAN),
            channel_width_hz=CHANNEL_WIDTH_HZ,
            field_delay_s=FIELD_DELAY_S,
        )
