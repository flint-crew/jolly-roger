from __future__ import annotations

import pickle
import shutil
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.coordinates import EarthLocation, SkyCoord

# Sesame-resolved position of PKSB1934-638, hardcoded so tests don't depend
# on network access to the CDS name resolver.
PKSB1934_638 = SkyCoord(ra=294.854277, dec=-63.712673, unit="deg")


@pytest.fixture(autouse=True)
def mock_skycoord_from_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def _from_name(name: str, *args: Any, **kwargs: Any) -> SkyCoord:
        if name == "PKSB1934-638":
            return PKSB1934_638
        msg = f"Unexpected name resolution request for {name!r} in tests"
        raise ValueError(msg)

    monkeypatch.setattr(SkyCoord, "from_name", staticmethod(_from_name))


@pytest.fixture
def ms_example(tmp_path: Path) -> Path:
    resource = files("jolly_roger.data").joinpath(
        "tests", "SB39400.RACS_0635-31.beam0.small.ms.zip"
    )

    outdir = tmp_path / "39400"

    with as_file(resource) as archive:
        shutil.unpack_archive(archive, outdir)

    return outdir / "SB39400.RACS_0635-31.beam0.small.ms"


@pytest.fixture
def ms_ant_xyz() -> np.ndarray:
    resource = files("jolly_roger.data").joinpath("tests", "ant_xyz.npz")
    with as_file(resource) as path, np.load(path) as arr:
        return arr["arr_0"]


@pytest.fixture
def ms_b_idx() -> np.ndarray:
    resource = files("jolly_roger.data").joinpath("tests", "b_idx.npz")
    with as_file(resource) as path, np.load(path) as arr:
        return arr["arr_0"]


@pytest.fixture
def ms_b_xyz() -> np.ndarray:
    resource = files("jolly_roger.data").joinpath("tests", "b_xyz.npz")
    with as_file(resource) as path, np.load(path) as arr:
        return arr["arr_0"]


@pytest.fixture
def ms_w_delays() -> np.ndarray:
    resource = files("jolly_roger.data").joinpath("tests", "w_delays.npz")
    with as_file(resource) as path, np.load(path) as arr:
        return arr["arr_0"]


@pytest.fixture
def ms_b_map() -> dict[Any, Any]:
    resource = files("jolly_roger.data").joinpath("tests", "b_map.pkl")
    with as_file(resource) as path, path.open("rb") as f:
        return dict(pickle.load(f))


@pytest.fixture
def ms_location() -> EarthLocation:
    return EarthLocation.from_geocentric(-2556146.7, 5097426.7, -2848333.2, unit="m")
