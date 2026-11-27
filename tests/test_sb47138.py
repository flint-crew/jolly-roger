"""Runs on 30 minutes of SB47138 around sunrise"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from casacore.tables import table

from jolly_roger.baselines import get_baselines
from jolly_roger.deconvolve import CleanTractorOptions, clean_tractor
from jolly_roger.hour_angles import get_location, make_hour_angles
from jolly_roger.tractor import TukeyTractorOptions, tukey_tractor
from jolly_roger.uvws import PhaseDirMismatchError, xyz_to_uvw


def test_sb47138_contents(ms_sb47138: Path) -> None:
    with table(str(ms_sb47138), ack=False) as tab:
        baselines = set(
            zip(tab.getcol("ANTENNA1"), tab.getcol("ANTENNA2"), strict=True)
        )
        n_times = len(np.unique(tab.getcol("TIME")))
        n_chan = tab.getcell("DATA", 0).shape[0]
    assert baselines == {(0, 3), (0, 5), (0, 6), (0, 31)}
    assert n_times == 181
    assert n_chan == 288


def test_sb47138_tukey(ms_sb47138: Path) -> None:
    tukey_tractor(
        ms_path=ms_sb47138,
        tukey_tractor_options=TukeyTractorOptions(output_column="NULLED_DATA"),
    )
    with table(str(ms_sb47138), ack=False) as tab:
        before, after = tab.getcol("DATA"), tab.getcol("NULLED_DATA")
        flagged = tab.getcol("FLAG").all(axis=(1, 2))
        ant_2 = tab.getcol("ANTENNA2")
    assert before.shape == after.shape
    assert not np.allclose(before, after)
    # The Sun crosses the field in delay on ak01-ak04 after sunrise
    # Every row with the Sun above the elevation cut: its notch overlaps the
    # field's protected range throughout
    assert flagged[ant_2 == 3].sum() == 120
    assert not flagged[ant_2 != 3].any()


def test_sb47138_ak01_ak04_contents(ms_sb47138_ak01_ak04: Path) -> None:
    with table(str(ms_sb47138_ak01_ak04), ack=False) as tab:
        baselines = set(
            zip(tab.getcol("ANTENNA1"), tab.getcol("ANTENNA2"), strict=True)
        )
        n_rows = tab.nrows()
    assert baselines == {(0, 3)}
    assert n_rows == 784


def test_sb47138_no_flag_overlap(ms_sb47138: Path) -> None:
    """With flag_overlap off, the notch still tapers the rows where the Sun
    meets the field, but does not flag them"""
    tukey_tractor(
        ms_path=ms_sb47138,
        tukey_tractor_options=TukeyTractorOptions(
            output_column="NULLED_DATA", flag_overlap=False
        ),
    )
    with table(str(ms_sb47138), ack=False) as tab:
        flagged = tab.getcol("FLAG").all(axis=(1, 2))
        before, after = tab.getcol("DATA"), tab.getcol("NULLED_DATA")
        ant_2 = tab.getcol("ANTENNA2")
    assert not flagged.any()
    assert not np.allclose(before[ant_2 == 3], after[ant_2 == 3])


def test_sb47138_clean_no_flag_overlap(ms_sb47138: Path) -> None:
    results = clean_tractor(
        ms_path=ms_sb47138,
        clean_tractor_options=CleanTractorOptions(
            output_column="CLEANED_DATA", flag_overlap=False
        ),
    )
    with table(str(ms_sb47138), ack=False) as tab:
        assert not tab.getcol("FLAG").all(axis=(1, 2)).any()
    assert results.statistics is not None
    assert results.statistics.rows_flagged > 0


def test_sb47138_rate_filter_recovers_rows(ms_sb47138_ak01_ak04: Path) -> None:
    """On 2 h of ak01-ak04 the Sun crosses the field in delay, but is separable
    in fringe rate: most of the rows the notch flags are recovered"""
    options = {"output_column": "NULLED_DATA", "elevation_cut_deg": -1.0}
    tukey_tractor(
        ms_path=ms_sb47138_ak01_ak04,
        tukey_tractor_options=TukeyTractorOptions(**options, rate_filter=True),
    )
    with table(str(ms_sb47138_ak01_ak04), ack=False) as tab:
        flagged = tab.getcol("FLAG").all(axis=(1, 2))
        before, after = tab.getcol("DATA"), tab.getcol("NULLED_DATA")
        with table(tab.getkeyword("HISTORY"), ack=False) as history:
            message = history.getcol("MESSAGE")[-1]
    assert message.startswith("Delay-rate filter: recovered 678 rows")
    assert flagged.sum() == 46
    # Recovered rows are changed, rows before sunrise are not
    assert not np.allclose(before[~flagged], after[~flagged])


@pytest.mark.parametrize("name", ["ms_sb47138", "ms_sb47138_ak01_ak04"])
def test_sb47138_field_matches_uvw(name: str, request: pytest.FixtureRequest) -> None:
    """FIELD PHASE_DIR is the direction the data and UVWs are phased to (beam 6,
    corrected as fixms does), not the boresight ASKAP writes"""
    ms = request.getfixturevalue(name)
    with table(str(ms), ack=False) as tab:
        ant_1, ant_2 = tab.getcol("ANTENNA1"), tab.getcol("ANTENNA2")
        uvw, time_centroid = tab.getcol("UVW"), tab.getcol("TIME_CENTROID")
    with table(str(ms / "ANTENNA"), ack=False) as tab:
        ant_xyz = tab.getcol("POSITION")
    with table(str(ms / "FIELD"), ack=False) as tab:
        phase_dir = SkyCoord(*tab.getcol("PHASE_DIR").squeeze(), unit="rad")

    baselines = get_baselines(ant_xyz)
    hour_angles = make_hour_angles(
        times_mjds=np.unique(time_centroid) * u.s,
        location=get_location(ant_xyz),
        position=phase_dir,
    )
    uvws = xyz_to_uvw(baselines=baselines, hour_angles=hour_angles).uvws.to(u.m).value
    baseline_idx = [
        baselines.b_map[(int(a1), int(a2))] for a1, a2 in zip(ant_1, ant_2, strict=True)
    ]
    time_idx = [hour_angles.time_map[t * u.s] for t in time_centroid]
    np.testing.assert_allclose(
        uvws[:, baseline_idx, time_idx].T, uvw, rtol=0, atol=1e-3
    )


def test_uncorrected_field_fails_early(ms_sb47138: Path) -> None:
    """With ASKAP's uncorrected FIELD (the boresight, kept by fixms as
    FIELD_OLD) the phase direction does not reproduce the UVW column"""
    with table(str(ms_sb47138 / "FIELD_OLD"), ack=False) as old:
        boresight = old.getcol("PHASE_DIR")
    with table(str(ms_sb47138 / "FIELD"), readonly=False, ack=False) as field:
        for column in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
            field.putcol(column, boresight)

    with pytest.raises(PhaseDirMismatchError, match="fix_ms_dir"):
        tukey_tractor(
            ms_path=ms_sb47138,
            tukey_tractor_options=TukeyTractorOptions(output_column="NULLED_DATA"),
        )
    # It fails before the measurement set is changed
    with table(str(ms_sb47138), ack=False) as tab:
        assert "NULLED_DATA" not in tab.colnames()
