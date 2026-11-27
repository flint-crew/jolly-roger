"""Runs on 30 minutes of SB47138 around sunrise"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from casacore.tables import table

from jolly_roger.deconvolve import CleanTractorOptions, clean_tractor
from jolly_roger.tractor import TukeyTractorOptions, tukey_tractor


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
    assert message.startswith("Delay-rate filter: recovered 701 rows")
    assert flagged.sum() == 23
    # Recovered rows are changed, rows before sunrise are not
    assert not np.allclose(before[~flagged], after[~flagged])
