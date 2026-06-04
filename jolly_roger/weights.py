"""Utility functions to help direct the rescaling of weight like
columns"""

from pathlib import Path

from casacore.tables import table

from jolly_roger.logging import logger



KNOWN_WEIGHTS = (
    "WEIGHT",
    "WEIGHT_SPECTRUM"
)

def find_weight_column(
    ms_path: Path
) -> str:
    logger.info("Searching for WEIGHT-like column")
    
    assert ms_path.exists(), f"MS file {ms_path} does not exist"
    
    weight_column: str | None = None
    
    with table(str(ms_path)) as tab:
        columns = tab.colnames()
        
    for col in KNOWN_WEIGHTS:
        if col in columns:
            weight_column = col
            break
        
    if weight_column is None:
        msg = f"Searched for {KNOWN_WEIGHTS=} but found none in {ms_path=}"
        raise ValueError(msg)

    logger.info(f"Found WEIGHT-like column {weight_column=}")
    return weight_column