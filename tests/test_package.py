from __future__ import annotations

import importlib.metadata

import jolly_roger as m


def test_version():
    assert importlib.metadata.version("jolly_roger") == m.__version__
