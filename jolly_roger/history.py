"""Record what was done to a measurement set in its HISTORY table"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from astropy.time import Time
from casacore.tables import table

from jolly_roger import __version__
from jolly_roger.logging import logger


def dataclass_to_app_params(options: Any) -> list[str]:
    """Format the fields of a dataclass as ``name = value`` strings for APP_PARAMS

    Args:
        options (Any): A dataclass instance

    Returns:
        list[str]: One string per field. Empty if ``options`` is not a dataclass.
    """
    if not is_dataclass(options):
        return []
    return [
        f"{field.name} = {getattr(options, field.name)!r}" for field in fields(options)
    ]


def add_history(
    ms_path: Path,
    messages: Sequence[str],
    origin: str,
    app_params: Sequence[str] | None = None,
) -> None:
    """Append one row per message to the HISTORY table of a measurement set.

    Args:
        ms_path (Path): The measurement set to record against
        messages (Sequence[str]): The messages to record, one row each
        origin (str): What produced the messages, e.g. ``jolly_tractor tukey``
        app_params (Sequence[str] | None, optional): Parameters used, recorded against each row. Defaults to None.
    """
    history_path = Path(ms_path) / "HISTORY"
    if not history_path.exists():
        logger.warning(f"{history_path} does not exist, not recording history")
        return

    params = list(app_params) if app_params else [""]
    cli_command = [" ".join(sys.argv)]
    now_mjd_s = Time.now().mjd * 86400.0

    with table(str(history_path), readonly=False, ack=False) as tab:
        start = tab.nrows()
        tab.addrows(len(messages))
        for offset, message in enumerate(messages):
            row = start + offset
            tab.putcell("TIME", row, now_mjd_s)
            tab.putcell("OBSERVATION_ID", row, -1)
            tab.putcell("MESSAGE", row, message)
            tab.putcell("PRIORITY", row, "INFO")
            tab.putcell("ORIGIN", row, origin)
            tab.putcell("OBJECT_ID", row, 0)
            tab.putcell("APPLICATION", row, f"jolly-roger {__version__}")
            tab.putcell("CLI_COMMAND", row, cli_command)
            tab.putcell("APP_PARAMS", row, params)

    logger.info(f"Recorded {len(messages)} HISTORY row(s) in {ms_path}")
