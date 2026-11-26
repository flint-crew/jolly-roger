from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.utils import iers

# Use the Earth orientation tables bundled with astropy (astropy-iers-data)
# so tests don't depend on network access to the IERS servers.
iers.conf.auto_download = False
iers.conf.auto_max_age = None

# Sesame-resolved position of PKSB1934-638, hardcoded so tests don't depend
# on network access to the CDS name resolver.
PKSB1934_638 = SkyCoord(ra=294.854277, dec=-63.712673, unit="deg")

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def mock_skycoord_from_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def _from_name(name: str) -> SkyCoord:
        if name == "PKSB1934-638":
            return PKSB1934_638
        msg = f"Unexpected name resolution request for {name!r} in tests"
        raise ValueError(msg)

    monkeypatch.setattr(SkyCoord, "from_name", staticmethod(_from_name))


@pytest.fixture
def ms_example(tmp_path: Path) -> Path:
    outdir = tmp_path / "39400"
    shutil.unpack_archive(DATA_DIR / "SB39400.RACS_0635-31.beam0.small.ms.zip", outdir)
    return outdir / "SB39400.RACS_0635-31.beam0.small.ms"


@pytest.fixture
def ms_sb47138(tmp_path: Path) -> Path:
    """30 minutes of SB47138 around sunrise: ak01 to ak04, ak06, ak07 and ak32"""
    shutil.unpack_archive(DATA_DIR / "sb47138_sunrise.ms.zip", tmp_path)
    return tmp_path / "sb47138_sunrise.ms"


@pytest.fixture
def ms_sb47138_ak01_ak04(tmp_path: Path) -> Path:
    """2 h 10 min of SB47138 ak01 to ak04 from just before sunrise, as the Sun
    crosses the field in delay"""
    shutil.unpack_archive(DATA_DIR / "sb47138_ak01_ak04.ms.zip", tmp_path)
    return tmp_path / "sb47138_ak01_ak04.ms"


@pytest.fixture
def ms_ant_xyz() -> np.ndarray:
    with np.load(DATA_DIR / "ant_xyz.npz") as arr:
        return arr["arr_0"]


@pytest.fixture
def ms_b_idx() -> np.ndarray:
    with np.load(DATA_DIR / "b_idx.npz") as arr:
        return arr["arr_0"]


@pytest.fixture
def ms_b_xyz() -> np.ndarray:
    with np.load(DATA_DIR / "b_xyz.npz") as arr:
        return arr["arr_0"]


@pytest.fixture
def ms_w_delays() -> np.ndarray:
    with np.load(DATA_DIR / "w_delays.npz") as arr:
        return arr["arr_0"]


@pytest.fixture
def ms_b_map() -> dict[Any, Any]:
    with (DATA_DIR / "b_map.pkl").open("rb") as f:
        return dict(pickle.load(f))


@pytest.fixture
def ms_location() -> EarthLocation:
    return EarthLocation.from_geocentric(-2556146.7, 5097426.7, -2848333.2, unit="m")
