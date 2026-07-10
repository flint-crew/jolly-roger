"""Tests around the tractor'ing. Some are simple, some are complex, but
all are important in their own way"""

from __future__ import annotations

from pathlib import Path

from casacore.tables import table

from jolly_roger.tractor import TukeyTractorOptions, tukey_tractor


def test_tractor_run1(ms_example) -> None:
    """A very simple end-to-end test identifying crashes"""

    new_column = "JACKS_DATA"

    tukey_tractor_options = TukeyTractorOptions(
        auto_size=True,
        guard_field=True,
        object_minimum_flux=0.05,
        output_column=new_column,
    )
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column not in cols

    tractor_results = tukey_tractor(
        ms_path=Path(ms_example),
        tukey_tractor_options=tukey_tractor_options,
    )

    assert tractor_results.output_plots is None
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column in cols


def test_tractor_run2(ms_example) -> None:
    """A very simple end-to-end test identifying crashes"""

    new_column = "JACKS_DATA"

    tukey_tractor_options = TukeyTractorOptions(
        auto_size=True,
        guard_field=True,
        object_minimum_flux=0.05,
        output_column=new_column,
        make_plots=True,
        number_of_plots=1,
    )
    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column not in cols

    tractor_results = tukey_tractor(
        ms_path=Path(ms_example),
        tukey_tractor_options=tukey_tractor_options,
    )

    assert tractor_results.output_plots is not None
    assert len(tractor_results.output_plots) == 1

    with table(str(ms_example), ack=False) as tab:
        cols = tab.colnames()
        assert new_column in cols
