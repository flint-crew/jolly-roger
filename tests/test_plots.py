from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from casacore.tables import table

from jolly_roger.baselines import BaselineData, get_baseline_data, get_open_ms_tables
from jolly_roger.plots import format_baseline_title


def _baseline_data(**kwargs: object) -> BaselineData:
    return BaselineData(
        masked_data=np.ma.masked_array(np.zeros((2, 3, 4))),
        freq_chan=np.linspace(1, 2, 3) * u.GHz,
        phase_center=SkyCoord(0, 0, unit="deg"),
        uvws_phase_center=np.zeros((3, 2)) * u.m,
        time=Time([60000.0, 60000.1], format="mjd"),
        ant_1=0,
        ant_2=6,
        **kwargs,  # type: ignore[arg-type]
    )


def test_format_baseline_title_indices_only() -> None:
    """Without names or a length, fall back to the antenna indices"""
    assert format_baseline_title(_baseline_data()) == "Ant 0 - Ant 6"


def test_format_baseline_title_names_and_length() -> None:
    """Antenna names and the baseline length are included when known"""
    data = _baseline_data(
        ant_1_name="ak01", ant_2_name="ak07", baseline_length=123.456 * u.m
    )
    assert format_baseline_title(data) == "ak01 - ak07 (Ant 0 - Ant 6), 123.5 m"


def test_format_baseline_title_long_baseline() -> None:
    """Long baselines are given in km"""
    data = _baseline_data(baseline_length=1234.5 * u.m)
    assert format_baseline_title(data) == "Ant 0 - Ant 6, 1.23 km"


def test_get_baseline_data_antenna_names_and_length(ms_example: Path) -> None:
    """Antenna names and the baseline length are read from the ANTENNA table"""
    with table(str(ms_example / "ANTENNA"), ack=False) as tab:
        names = tab.getcol("NAME")
        positions = tab.getcol("POSITION")

    with get_open_ms_tables(ms_path=ms_example) as open_ms_tables:
        data = get_baseline_data(open_ms_tables=open_ms_tables, ant_1=0, ant_2=6)

    assert data.ant_1_name == names[0]
    assert data.ant_2_name == names[6]
    assert data.baseline_length is not None
    assert np.isclose(
        data.baseline_length.to("m").value,
        np.linalg.norm(positions[0] - positions[6]),
    )
    assert format_baseline_title(data).startswith(f"{names[0]} - {names[6]} (")
