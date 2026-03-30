from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from numpy import ma

from jolly_roger.delays import data_to_delay_time, delay_time_to_data
from jolly_roger.tractor import DataChunk


def make_data_chunk(
    n_time: int = 8,
    n_chan: int = 16,
    n_pol: int = 2,
    mask_fraction: float = 0.0,
    seed: int = 42,
) -> DataChunk:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_time, n_chan, n_pol)) + 1j * rng.standard_normal(
        (n_time, n_chan, n_pol)
    )
    mask = np.zeros_like(data, dtype=bool)
    if mask_fraction > 0.0:
        mask = rng.random(data.shape) < mask_fraction

    freq_chan = np.linspace(1.0, 2.0, n_chan) * u.GHz

    return DataChunk(
        masked_data=ma.masked_array(data, mask=mask),
        freq_chan=freq_chan,
        phase_center=SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg),
        uvws_phase_center=np.zeros((n_time, 3)) * u.m,
        time=Time.now(),
        time_mjds=np.zeros(n_time),
        ant_1=np.zeros(n_time, dtype=np.int64),
        ant_2=np.ones(n_time, dtype=np.int64),
        row_start=0,
        chunk_size=n_time,
    )


def test_unmasked_data_is_recovered():
    """Unmasked data should survive a forward + inverse FFT unchanged."""
    data = make_data_chunk()
    original_data = data.masked_data.data.copy()

    delay_time = data_to_delay_time(data)
    recovered_data = delay_time_to_data(delay_time, data)

    np.testing.assert_allclose(
        recovered_data.masked_data.data,
        original_data,
        atol=1e-10,
        err_msg="Round-trip introduced unexpected numerical error",
    )


def test_mask_is_preserved():
    """The original mask must be reapplied after the round-trip."""
    data = make_data_chunk(mask_fraction=0.1)
    original_mask = data.masked_data.mask.copy()

    delay_time = data_to_delay_time(data)
    recovered_data = delay_time_to_data(delay_time, data)

    np.testing.assert_array_equal(
        recovered_data.masked_data.mask,
        original_mask,
        err_msg="Mask was altered during round-trip",
    )


def test_masked_channels_zero_filled_in_forward_pass():
    """data_to_delay_time fills masked values with 0+0j before FFT."""
    data = make_data_chunk(mask_fraction=0.2)

    delay_time = data_to_delay_time(data)

    assert np.all(np.isfinite(delay_time.delay_time)), (
        "delay_time contains non-finite values after zero-filling masked data"
    )


def test_delay_units_and_shape():
    """Delay array must have time units and match the channel axis length."""
    n_chan = 32
    data = make_data_chunk(n_chan=n_chan)

    delay_time = data_to_delay_time(data)

    assert delay_time.delay.unit.physical_type == "time", (
        f"Expected delay to have time units, got {delay_time.delay.unit}"
    )
    assert len(delay_time.delay) == n_chan, (
        f"Expected {n_chan} delay bins, got {len(delay_time.delay)}"
    )


def test_output_shape_unchanged():
    """The recovered_data DataChunk must have the same data shape as the input."""
    n_time, n_chan, n_pol = 6, 20, 4
    data = make_data_chunk(n_time=n_time, n_chan=n_chan, n_pol=n_pol)
    expected_shape = data.masked_data.shape

    delay_time = data_to_delay_time(data)
    recovered_data = delay_time_to_data(delay_time, data)

    assert recovered_data.masked_data.shape == expected_shape, (
        f"Shape changed during round-trip: {expected_shape} → {recovered_data.masked_data.shape}"
    )


def test_metadata_unchanged():
    """Scalar metadata fields on DataChunk must be untouched by the round-trip."""
    data = make_data_chunk()
    original_freq = data.freq_chan.copy()
    original_row_start = data.row_start
    original_chunk_size = data.chunk_size

    delay_time = data_to_delay_time(data)
    recovered_data = delay_time_to_data(delay_time, data)

    assert u.allclose(recovered_data.freq_chan, original_freq), "freq_chan was mutated"
    assert recovered_data.row_start == original_row_start, "row_start was mutated"
    assert recovered_data.chunk_size == original_chunk_size, "chunk_size was mutated"


@pytest.mark.parametrize("n_chan", [8, 16, 64, 128])
def test_round_trip_various_channel_counts(n_chan: int):
    """Round-trip accuracy must hold across different channel counts."""
    data = make_data_chunk(n_chan=n_chan)
    original_data = data.masked_data.data.copy()

    delay_time = data_to_delay_time(data)
    recovered_data = delay_time_to_data(delay_time, data)

    np.testing.assert_allclose(
        recovered_data.masked_data.data,
        original_data,
        atol=1e-10,
        err_msg=f"Round-trip failed for n_chan={n_chan}",
    )
