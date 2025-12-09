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

<img src="logo.png" alt="The Jolly Roger Flag" style="width:400px;"/>


## Installation

`pip install jolly-roger`

## About

This package attempts to flag or modify visibilities that are contaminated by the Sun (or potentially some other bright source). There are two main modes that are currently supported in `jolly-roger`.

The flags based on the projected baseline length should the array be tracking the Sun. The projected baseline length between some phased direction being tracked and the Sun can be significantly different. `jolly-roger` attempts to leverage this by only flagging data where the projected baseline length is between some nominal range that corresponds to angular scales associated with the Sun.

The second mode is the application of a notch filter in delay space, where the nulling is applied to the expected delay of the Sun for each timestep and baseline. This mode reads the data row-wise as it appears in the measurement set, so can be fast enough to apply to long-tracks ASKAP observations while also using a limited set of computing resources. A ten hour ASKAP continuum observation can be processed in ~2 minutes, with the process being mostly I/O bound.

`jolly-roger` makes no guarentess about removing all contaminated visibilities, nor does it attempt to peel/subtract the Sun from the visibility data.

## Usage

For the flagger run:

```
jolly_flagger --help
```

For the nulling filter run:

```
jolly_tractor --help
```

Details are provide in the [docs][rtd-link].
