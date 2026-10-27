"""Tests for recording processing in the HISTORY table"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from casacore.tables import table

from jolly_roger.history import add_history, dataclass_to_app_params


@dataclass
class _Options:
    width_ns: float = 10.0
    name: str = "sun"


def test_dataclass_to_app_params() -> None:
    """Fields become name = value strings; anything else gives nothing"""
    assert dataclass_to_app_params(_Options()) == ["width_ns = 10.0", "name = 'sun'"]
    assert dataclass_to_app_params({"width_ns": 10.0}) == []


def test_add_history_rows(ms_example: Path) -> None:
    """One row per message, with the origin and parameters recorded"""
    with table(str(ms_example / "HISTORY"), ack=False) as tab:
        before = tab.nrows()

    add_history(
        ms_path=ms_example,
        messages=["first", "second"],
        origin="test",
        app_params=["a = 1"],
    )

    with table(str(ms_example / "HISTORY"), ack=False) as tab:
        assert tab.nrows() == before + 2
        assert tab.getcell("MESSAGE", before) == "first"
        assert tab.getcell("MESSAGE", before + 1) == "second"
        assert tab.getcell("ORIGIN", before + 1) == "test"
        assert list(tab.getcell("APP_PARAMS", before + 1)) == ["a = 1"]


def test_add_history_without_history_table(tmp_path: Path) -> None:
    """A measurement set without a HISTORY table is left alone"""
    add_history(ms_path=tmp_path, messages=["ignored"], origin="test")
    assert not (tmp_path / "HISTORY").exists()
