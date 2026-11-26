"""Runs on 30 minutes of SB47138 around sunrise"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from casacore.tables import table

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
    assert flagged[ant_2 == 3].sum() == 83
    assert not flagged[ant_2 != 3].any()


def test_sb47138_ak01_ak04_contents(ms_sb47138_ak01_ak04: Path) -> None:
    with table(str(ms_sb47138_ak01_ak04), ack=False) as tab:
        baselines = set(
            zip(tab.getcol("ANTENNA1"), tab.getcol("ANTENNA2"), strict=True)
        )
        n_rows = tab.nrows()
    assert baselines == {(0, 3)}
    assert n_rows == 784
