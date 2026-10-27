"""Tests for removing objects outside the field by CLEAN in delay"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from capn_crunch import create_options_from_parser
from casacore.tables import table
from numpy import ma
from numpy.typing import NDArray

from jolly_roger.deconvolve import (
    CleanTractorOptions,
    DelayCleanOptions,
    DelayCleanStatistics,
    MissingCleanDependencyError,
    clean_tractor,
    compute_clean_tractor,
    compute_delay_clean,
    delay_clean_spectrum,
    describe_band_edges,
    describe_clean_line_limit,
    describe_clean_run,
    estimate_visibility_noise,
)
from jolly_roger.tractor import DataChunk, cli, get_parser
from jolly_roger.uvws import WDelays

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
    data_chunk = _chunk([150e-9, 55e-9, 150e-9])
    # A fully flagged row is passed through unchanged
    data_chunk.masked_data.mask[2] = True
    flagged_row = data_chunk.masked_data.data[2].copy()

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
    assert result.flags[2].all()
    np.testing.assert_array_equal(cleaned.data[2], flagged_row)


def test_missing_rm_lite_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rm_lite.utils", None)

    with pytest.raises(MissingCleanDependencyError, match=r"jolly-roger\[clean\]"):
        delay_clean_spectrum(
            values=_field(),
            weights=np.ones(N_CHAN),
            channel_width_hz=CHANNEL_WIDTH_HZ,
            field_delay_s=FIELD_DELAY_S,
        )


def _w_delays(
    object_delay_s: float, n_rows: int, elevation_deg: list[float], guard_s: float = 0.0
) -> WDelays:
    """One baseline (0, 1) whose object sits at a fixed delay, one row per time"""
    return WDelays(
        object_name="sun",
        w_delays=np.full((1, n_rows), object_delay_s) * u.s,
        b_map={(0, 1): 0},
        time_map={float(t) * u.s: t for t in range(n_rows)},
        elevation=np.array(elevation_deg) * u.deg,
        guard_region=np.full((1, n_rows), guard_s) * u.s if guard_s else None,
    )


def test_compute_clean_tractor_cleans_only_rows_with_the_object_up() -> None:
    data_chunk = _chunk([150e-9, 150e-9])
    original = data_chunk.masked_data.data.copy()
    w_delays = _w_delays(150e-9, n_rows=2, elevation_deg=[20.0, -10.0])

    result = compute_clean_tractor(
        data_chunk=data_chunk,
        clean_tractor_options=CleanTractorOptions(
            field_delay_ns=FIELD_DELAY_S * 1e9, visibility_noise=SIGMA
        ),
        w_delays_list=[w_delays],
    )

    assert result.data_chunk is not None
    cleaned = result.data_chunk.masked_data.data
    # The object is above the elevation cut in row 0 only
    assert _rms(cleaned[0, :, 0] - original[0, :, 0]) > 1.0
    np.testing.assert_array_equal(cleaned[1], original[1])


def test_compute_clean_tractor_widens_the_field_by_the_guard() -> None:
    """With a guard band reaching past the object, the object is kept. Only its
    faint wings beyond the guard band are subtracted."""
    data_chunk = _chunk([150e-9])
    w_delays = _w_delays(150e-9, n_rows=1, elevation_deg=[20.0], guard_s=200e-9)

    result = compute_clean_tractor(
        data_chunk=data_chunk,
        clean_tractor_options=CleanTractorOptions(
            field_delay_ns=FIELD_DELAY_S * 1e9, visibility_noise=SIGMA
        ),
        w_delays_list=[w_delays],
    )

    assert result.data_chunk is not None
    subtracted = result.data_chunk.masked_data.data - data_chunk.masked_data.data
    # The object has rms 10
    assert _rms(subtracted) < 0.1


def test_compute_clean_tractor_skips_autocorrelations() -> None:
    data_chunk = replace(_chunk([150e-9]), ant_2=np.zeros(1, dtype=np.int64))
    w_delays = _w_delays(150e-9, n_rows=1, elevation_deg=[20.0])

    result = compute_clean_tractor(
        data_chunk=data_chunk,
        clean_tractor_options=CleanTractorOptions(visibility_noise=SIGMA),
        w_delays_list=[w_delays],
    )

    assert result.nothing_to_do


def test_statistics_and_descriptions() -> None:
    statistics = DelayCleanStatistics()
    compute_delay_clean(
        data_chunk=_chunk([150e-9, 55e-9]),
        field_delay_s=FIELD_DELAY_S,
        visibility_noise=SIGMA,
        statistics=statistics,
    )
    options = CleanTractorOptions(target_objects=("sun", "moon"))

    assert statistics.spectra == 4
    assert statistics.rows_flagged == 1
    assert len(statistics.edge_residuals) == 4

    run = describe_clean_run(statistics, options)
    assert "4 spectra where sun, moon was above the elevation cut" in run
    assert "1 rows were flagged" in run

    band_edges = describe_band_edges(statistics)
    assert band_edges is not None
    assert "8 channels at each end of the band" in band_edges
    assert describe_band_edges(DelayCleanStatistics()) is None

    # 3 sqrt(288) / 1.06 = 48
    assert "48/n times the noise" in describe_clean_line_limit(288, 3.0)


def test_statistics_count_spectra_without_a_noise_estimate() -> None:
    data_chunk = _chunk([150e-9])
    data_chunk.masked_data.mask[:, ::20, :] = True
    statistics = DelayCleanStatistics()

    compute_delay_clean(
        data_chunk=data_chunk, field_delay_s=FIELD_DELAY_S, statistics=statistics
    )

    assert statistics.noise_not_estimated == 2
    assert "2 spectra were left unchanged" in describe_clean_run(
        statistics, CleanTractorOptions()
    )


def test_short_spectrum_has_no_band_edge_residual() -> None:
    weights = np.zeros(N_CHAN)
    weights[:20] = 1.0

    result = delay_clean_spectrum(
        values=_field() + _noise(7),
        weights=weights,
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=FIELD_DELAY_S,
        visibility_noise=SIGMA,
    )

    assert result.edge_residual is None
    assert result.mid_residual is None


def test_clean_tractor_records_history(ms_example: Path) -> None:
    # The test measurement set is fully flagged, so unflag it to have CLEAN run
    with table(str(ms_example), readonly=False, ack=False) as tab:
        tab.putcol("FLAG", np.zeros_like(tab.getcol("FLAG")))
    with table(str(Path(ms_example) / "HISTORY"), ack=False) as tab:
        before = tab.nrows()

    result = clean_tractor(
        ms_path=Path(ms_example),
        clean_tractor_options=CleanTractorOptions(
            output_column="CLEANED_DATA",
            visibility_noise=1.0,
            # Clean every row, wherever the Sun is in this observation
            elevation_cut_deg=-90.0,
            ignore_nyquist_zone=1000,
        ),
    )

    assert result.statistics is not None
    assert result.statistics.spectra > 0
    with table(str(ms_example), ack=False) as tab:
        assert "CLEANED_DATA" in tab.colnames()
    with table(str(Path(ms_example) / "HISTORY"), ack=False) as tab:
        assert tab.nrows() > before
        assert set(tab.getcol("ORIGIN")[before:]) == {"jolly_tractor clean"}


def test_clean_tractor_makes_plots(ms_example: Path) -> None:
    result = clean_tractor(
        ms_path=Path(ms_example),
        clean_tractor_options=CleanTractorOptions(make_plots=True, number_of_plots=1),
    )

    assert result.output_plots
    assert all(path.exists() for path in result.output_plots)


def test_clean_subcommand_options() -> None:
    args = get_parser().parse_args(
        ["clean", "example.ms", "--visibility-noise", "0.5", "--field-delay-ns", "20"]
    )
    options = create_options_from_parser(
        parser_namespace=args, options_class=CleanTractorOptions
    )

    assert args.mode == "clean"
    assert options.visibility_noise == 0.5
    assert options.field_delay_ns == 20
    # The guard band is on unless the flag turns it off
    assert options.guard_field
    args = get_parser().parse_args(["clean", "example.ms", "--guard-field"])
    assert not create_options_from_parser(
        parser_namespace=args, options_class=CleanTractorOptions
    ).guard_field


def test_cli_runs_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        "jolly_roger.deconvolve.clean_tractor",
        lambda ms_path, clean_tractor_options: calls.append(
            (ms_path, clean_tractor_options)
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["jolly_tractor", "clean", "example.ms", "--mask-sigma", "6"]
    )

    cli()

    assert len(calls) == 1
    ms_path, options = calls[0]
    assert ms_path == Path("example.ms")
    assert options.mask_sigma == 6
