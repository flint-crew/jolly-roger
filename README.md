# jolly-roger

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
<!-- [![Conda-Forge][conda-badge]][conda-link] -->
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/flint-crew/jolly-roger/workflows/CI/badge.svg
[actions-link]:             https://github.com/flint-crew/jolly-roger/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/jolly-roger
[conda-link]:               https://github.com/conda-forge/jolly-roger-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/flint-crew/jolly-roger/discussions
[pypi-link]:                https://pypi.org/project/jolly-roger/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/jolly-roger
[pypi-version]:             https://img.shields.io/pypi/v/jolly-roger
[rtd-badge]:                https://readthedocs.org/projects/jolly-roger/badge/?version=latest
[rtd-link]:                 https://jolly-roger.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

The pirate flagger!

# About

Calculate the (u,v,w)'s towards the Sun for some measurement set and flag based on the projected baseline, i.e. the uv-distance.

```
jolly_flagger scienceData.EMU_1141-55.SB47138.EMU_1141-55.beam00_averaged_cal.leakage.ms --min-horizon-limit-deg '-3' --max-horizon-limit-deg 30 --max-scale-deg 10
```
