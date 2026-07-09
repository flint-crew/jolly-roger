from pathlib import Path
from typing import Any

import numpy as np

from jolly_roger.baselines import get_baselines_from_ms
from jolly_roger.hour_angles import get_location_from_ms, make_hour_angles_for_ms
from jolly_roger.uvws import get_object_delay_for_ms


def test_baselines(
        ms_example: Path, 
        ms_ant_xyz: np.ndarray,
        ms_b_idx: np.ndarray,
        ms_b_xyz: np.ndarray,
        ms_b_map: dict[Any, Any],
) -> None:
    baselines = get_baselines_from_ms(ms_path=ms_example)
    np.testing.assert_array_equal(baselines.ant_xyz.value, ms_ant_xyz)
    np.testing.assert_array_equal(baselines.b_idx, ms_b_idx)
    np.testing.assert_array_equal(baselines.b_xyz.value, ms_b_xyz)
    for key, val in baselines.b_map.items():
        assert ms_b_map[key] == val
