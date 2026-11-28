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
    _object_band_mask,
    clean_tractor,
    compute_clean_tractor,
    compute_delay_clean,
    delay_axis,
    delay_clean_spectra,
    delay_clean_spectrum,
    describe_band_edges,
    describe_clean_line_limit,
    describe_clean_run,
    estimate_visibility_noise,
    protected_channel_mask,
)
from jolly_roger.lines import KAPPA
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


def test_old_rm_lite_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    from rm_lite.utils import clean  # noqa: PLC0415

    monkeypatch.delattr(clean, "minor_loop_batch")

    with pytest.raises(MissingCleanDependencyError, match=r"rm-lite>=2026\.11\.0"):
        delay_clean_spectrum(
            values=_field(),
            weights=np.ones(N_CHAN),
            channel_width_hz=CHANNEL_WIDTH_HZ,
            field_delay_s=FIELD_DELAY_S,
        )


def _mixed_chunk() -> DataChunk:
    """Rows with different flag patterns, objects inside and outside the
    field, a fully flagged spectrum, and spectra too broken up by flags for the
    noise to be estimated"""
    data_chunk = _chunk([150e-9, 55e-9, 150e-9, -120e-9, 200e-9, 150e-9])
    mask = data_chunk.masked_data.mask
    mask[1, 90:100, 0] = True
    mask[3, :10, :] = True
    mask[4, :, 1] = True
    mask[5, ::20, :] = True
    return data_chunk


def test_chunk_matches_cleaning_each_spectrum() -> None:
    data_chunk = _mixed_chunk()
    statistics = DelayCleanStatistics()

    result = compute_delay_clean(
        data_chunk=data_chunk, field_delay_s=FIELD_DELAY_S, statistics=statistics
    )

    data = data_chunk.masked_data.data
    flags = ma.getmaskarray(data_chunk.masked_data)
    expected = data.copy()
    reaches_field = np.zeros(data.shape[0], dtype=bool)
    expected_statistics = DelayCleanStatistics()
    spectra = []
    for row in range(data.shape[0]):
        for pol in range(data.shape[2]):
            weights = (~flags[row, :, pol]).astype(float)
            if not weights.any():
                continue
            spectrum = delay_clean_spectrum(
                values=data[row, :, pol],
                weights=weights,
                channel_width_hz=CHANNEL_WIDTH_HZ,
                field_delay_s=FIELD_DELAY_S,
            )
            expected[row, :, pol] -= spectrum.model
            reaches_field[row] |= spectrum.reaches_field
            spectra.append(spectrum)
    expected_statistics.record(spectra, rows_flagged=int(reaches_field.sum()))
    expected_flags = flags | reaches_field[:, None, None]

    assert result.data_chunk is not None
    assert result.flags is not None
    np.testing.assert_allclose(
        result.data_chunk.masked_data.data, expected, rtol=0, atol=1e-10
    )
    np.testing.assert_array_equal(result.flags, expected_flags)
    # Both kinds of uncleaned spectra are exercised
    assert statistics.spectra == len(spectra) == 11
    assert statistics.noise_not_estimated == 2
    assert statistics.noise_not_estimated == expected_statistics.noise_not_estimated
    np.testing.assert_allclose(
        statistics.edge_residuals, expected_statistics.edge_residuals
    )
    np.testing.assert_allclose(
        statistics.mid_residuals, expected_statistics.mid_residuals
    )
    assert statistics.rows_flagged == expected_statistics.rows_flagged == 1


def test_spectra_match_cleaning_each_spectrum() -> None:
    values = np.stack(
        [_field() + _noise(seed) + _resolved_object(150e-9) for seed in range(3)]
    )
    weights = np.ones(N_CHAN)
    weights[40:46] = 0.0
    # The last spectrum's field takes in the object
    field_delays = np.array([FIELD_DELAY_S, FIELD_DELAY_S, 300e-9])
    noise = np.array([SIGMA, np.nan, 2 * SIGMA])

    results = delay_clean_spectra(
        values=values,
        weights=weights,
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=field_delays,
        visibility_noise=noise,
    )

    for index, result in enumerate(results):
        expected = delay_clean_spectrum(
            values=values[index],
            weights=weights,
            channel_width_hz=CHANNEL_WIDTH_HZ,
            field_delay_s=float(field_delays[index]),
            visibility_noise=None if np.isnan(noise[index]) else float(noise[index]),
        )
        if index == 1:
            # A NaN noise is not known, so the spectrum is left alone
            assert result.iterations == 0
            assert result.delay_noise == 0
            assert np.all(result.model == 0)
            continue
        np.testing.assert_allclose(result.model, expected.model, rtol=0, atol=1e-12)
        assert result.iterations == expected.iterations
        assert result.delay_noise == pytest.approx(expected.delay_noise)
        assert result.reaches_field == expected.reaches_field
        assert result.edge_residual == pytest.approx(expected.edge_residual)
        assert result.mid_residual == pytest.approx(expected.mid_residual)
    assert _rms(results[0].model) > 1.0
    assert _rms(results[2].model) < 0.1


def test_chunk_results_do_not_depend_on_row_order() -> None:
    data_chunk = _mixed_chunk()
    order = np.array([4, 2, 0, 5, 1, 3])
    shuffled = replace(
        data_chunk,
        masked_data=data_chunk.masked_data[order],
        ant_1=data_chunk.ant_1[order],
        ant_2=data_chunk.ant_2[order],
        time_mjds=data_chunk.time_mjds[order],
    )

    result = compute_delay_clean(data_chunk=data_chunk, field_delay_s=FIELD_DELAY_S)
    shuffled_result = compute_delay_clean(
        data_chunk=shuffled, field_delay_s=FIELD_DELAY_S
    )

    assert result.data_chunk is not None
    assert shuffled_result.data_chunk is not None
    assert result.flags is not None
    assert shuffled_result.flags is not None
    np.testing.assert_allclose(
        shuffled_result.data_chunk.masked_data.data,
        result.data_chunk.masked_data.data[order],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(shuffled_result.flags, result.flags[order])


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

    # mask_sigma sqrt(288) / 1.06 = 80 for mask_sigma = 5
    assert "80/n times the noise" in describe_clean_line_limit(288, 5.0)


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
    # The guard band is on unless turned off
    assert options.guard_field
    args = get_parser().parse_args(["clean", "example.ms", "--no-guard-field"])
    assert not create_options_from_parser(
        parser_namespace=args, options_class=CleanTractorOptions
    ).guard_field
    args = get_parser().parse_args(["clean", "example.ms", "--rate-filter"])
    assert create_options_from_parser(
        parser_namespace=args, options_class=CleanTractorOptions
    ).rate_filter


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


def _line(peak: float, fwhm_channels: float, channel: int = 64) -> NDArray[np.floating]:
    """A Gaussian spectral line from a source at the phase centre"""
    return peak * np.exp(
        -4 * np.log(2) * ((np.arange(N_CHAN) - channel) / fwhm_channels) ** 2
    )


def _line_loss(
    object_delay_s: float | None, half_width_s: float, fwhm_channels: float
) -> float:
    """Fraction of a bright line's peak subtracted along with a point object at
    150 ns, with only the object's band subtracted, or everything outside the
    field if ``object_delay_s`` is None"""
    peak = 1e4 * SIGMA
    point = 100 * _fringe(150e-9)
    (result,) = delay_clean_spectra(
        values=(_line(peak, fwhm_channels) + point + _noise(3))[None],
        weights=np.ones(N_CHAN),
        channel_width_hz=CHANNEL_WIDTH_HZ,
        field_delay_s=FIELD_DELAY_S,
        options=DelayCleanOptions(object_margin_resolutions=0.0),
        visibility_noise=SIGMA,
        object_delay_s=None if object_delay_s is None else np.array([[object_delay_s]]),
        object_extent_s=np.array([[half_width_s]]),
    )
    return float(np.real(result.model[64] - point[64]) / peak)


def test_line_loss_matches_the_analytic_form() -> None:
    """A bright line in the field loses kappa dnu 2W exp(-(pi dnu tau)^2 / 4 ln 2)
    to an object band of half-width W at delay tau, 2 W dnu when it is narrower
    than a channel; subtracting everything outside the field it loses
    1 - 2 tau_f dnu"""
    half_width_s = 20e-9
    assert _line_loss(150e-9, half_width_s, 0.5) == pytest.approx(
        2 * half_width_s * CHANNEL_WIDTH_HZ, rel=0.2
    )
    dnu = 3 * CHANNEL_WIDTH_HZ
    expected = (
        KAPPA
        * dnu
        * 2
        * half_width_s
        * np.exp(-((np.pi * dnu * 150e-9) ** 2) / (4 * np.log(2)))
    )
    assert _line_loss(150e-9, half_width_s, 3) == pytest.approx(expected, abs=0.01)
    assert _line_loss(None, 0.0, 0.5) == pytest.approx(
        1 - 2 * FIELD_DELAY_S * CHANNEL_WIDTH_HZ, rel=0.05
    )


def test_object_band_wraps_and_skips_inactive_objects() -> None:
    period_s = 1 / CHANNEL_WIDTH_HZ
    delays_s = delay_axis(N_CHAN, CHANNEL_WIDTH_HZ, 4)
    keep, meets_field = _object_band_mask(
        delays_s=delays_s,
        # Object 1 is one period away (a later Nyquist zone); object 2 is inactive
        object_delay_s=np.array([[150e-9, np.nan], [150e-9 + period_s, np.nan]]),
        object_halfwidth_s=np.full((2, 2), 20e-9),
        field_delay_s=np.full(2, FIELD_DELAY_S),
        period_s=period_s,
    )
    np.testing.assert_array_equal(keep[0], keep[1])
    np.testing.assert_array_equal(keep[0], np.abs(delays_s - 150e-9) <= 20e-9)
    assert not meets_field.any()

    _, meets_field = _object_band_mask(
        delays_s=delays_s,
        object_delay_s=np.array([[45e-9]]),
        object_halfwidth_s=np.array([[20e-9]]),
        field_delay_s=np.array([FIELD_DELAY_S]),
        period_s=period_s,
    )
    assert meets_field.all()


def test_compute_clean_tractor_subtracts_only_the_object_band() -> None:
    """Emission outside the field but away from the object is kept, and the
    band widens by the primary beam's D / c"""
    data_chunk = _chunk([150e-9])
    other = 3 * _fringe(-200e-9)
    data_chunk.masked_data.data[0] += other[:, None]
    w_delays = _w_delays(150e-9, n_rows=1, elevation_deg=[20.0])
    options = CleanTractorOptions(
        field_delay_ns=FIELD_DELAY_S * 1e9, visibility_noise=SIGMA
    )

    result = compute_clean_tractor(
        data_chunk=data_chunk,
        clean_tractor_options=options,
        w_delays_list=[w_delays],
        # The object is 40 ns across with a 25 ns ripple in its spectrum, which
        # the band must cover as it would a real object's beam-induced spread
        beam_delay_s=80e-9,
    )

    assert result.data_chunk is not None
    cleaned = result.data_chunk.masked_data
    expected = _field() + _noise(0) + other
    good = ~cleaned.mask[0, :, 0]
    assert (
        np.median(np.abs((cleaned.data[0, :, 0] - expected)[good])) < 3 * NOISE_MEDIAN
    )


def test_protected_channels_keep_a_line() -> None:
    np.testing.assert_array_equal(
        protected_channel_mask(FREQ_HZ, (1210.0, 1214.0)),
        (FREQ_HZ >= 1210e6) & (FREQ_HZ <= 1214e6),
    )
    assert protected_channel_mask(FREQ_HZ, None) is None
    with pytest.raises(ValueError, match="pairs"):
        protected_channel_mask(FREQ_HZ, (1210.0,))

    line = _line(1e4 * SIGMA, 1)
    data_chunk = _chunk([150e-9])
    data_chunk.masked_data.data[0] += line[:, None]
    protected = protected_channel_mask(
        FREQ_HZ, (FREQ_HZ[62] * 1e-6, FREQ_HZ[66] * 1e-6)
    )
    result = compute_delay_clean(
        data_chunk=data_chunk,
        field_delay_s=FIELD_DELAY_S,
        visibility_noise=SIGMA,
        object_delay_s=np.array([[150e-9]]),
        object_extent_s=np.array([[60e-9]]),
        protected_channels=protected,
    )

    assert result.data_chunk is not None
    cleaned = result.data_chunk.masked_data.data[0, :, 0]
    expected = _field() + _noise(0) + line
    # The line is kept to the noise; the object is still removed beneath it
    assert abs(cleaned[64] - expected[64]) < 10 * SIGMA


def test_clean_line_options() -> None:
    args = get_parser().parse_args(
        [
            "clean",
            "example.ms",
            "--target-objects",
            "sun",
            "Cyg A",
            "--object-radius-deg",
            "0.5",
            "0.05",
            "--protect-mhz",
            "1420",
            "1421",
        ]
    )
    options = create_options_from_parser(
        parser_namespace=args, options_class=CleanTractorOptions
    )
    assert options.target_objects == ("sun", "Cyg A")
    assert options.object_radius_deg == (0.5, 0.05)
    assert options.protect_mhz == (1420.0, 1421.0)
