"""Tests for the sinc response utilieis"""

from __future__ import annotations

import astropy.units as u
import numpy as np

from jolly_roger.response import (
    calculate_expected_sinc_width,
    get_delay_of_nth_sidelobe,
)

# Example code used to investigate and create tests
"""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

no_chans = 288
freqs = (np.arange(no_chans) + 800) * u.MHz
bandpass = np.ones(no_chans)


diff = (1 / (freqs.max() - freqs.min())).decompose().to("s").value

fft_band = np.fft.fftshift(
    np.fft.fft(bandpass, norm="backward")
)
fft_delay = np.fft.fftshift(
    np.fft.fftfreq(no_chans, np.diff(freqs)[0])
)

offt_band = np.fft.fftshift(
    np.fft.fft(bandpass, 10028, norm="backward"),
)
offt_delay = np.fft.fftshift(
    np.fft.fftfreq(10028, np.diff(freqs)[0])
)


fig, ax = plt.subplots(1,1)


ax.plot(
    offt_delay.to("s").value,
    np.abs(offt_band),
    label="Top hat"
)

ax.plot(
    fft_delay.to("s").value,
    np.abs(fft_band),
    label="Padded Tophat"
)

ax.axvline(-diff, color="black", label="Main lobe")
ax.axvline(diff, color="black")

ax.set(
    xlim=[-0.5e-7, 0.5e-7],
    xlabel="Delay / s",
    ylabel="Power"
)

for i in range(1, 10):
    space = diff * (i + 0.5)
    ax.axvline(space, label="Sidelobe" if i==1 else None)
    ax.axvline(-space)


ax.legend()
"""


def test_calculate_sinc_width() -> None:
    """Calculate the expected sinc response width in delay space from
    a set of known frequencies"""

    no_chans = 288
    freqs = (np.arange(no_chans) + 800) * u.MHz

    width = calculate_expected_sinc_width(freqs=freqs)

    assert width == (3.4843205574912892e-09 * u.s)


def test_calculate_sinc_width_no_unit() -> None:
    """Calculate the expected sinc response width in delay space from
    a set of known frequencies. Same input as above but do not attach
    frequency information to input"""

    no_chans = 288
    freqs_mhz = np.arange(no_chans) + 800

    width = calculate_expected_sinc_width(freqs=freqs_mhz)

    assert width == (3.4843205574912892e-09 * u.s)


def test_nth_sidelobe() -> None:
    """Ensure that an appropriate sidelobe position can be obtained"""

    width = 3.4843205574912892e-09 * u.s
    nth_sidelobe_delay = get_delay_of_nth_sidelobe(sinc_width=width, n=1)

    assert nth_sidelobe_delay == (5.226480836236934e-09 * u.s)

    nth_sidelobe_delay = get_delay_of_nth_sidelobe(sinc_width=width, n=4)
    assert nth_sidelobe_delay == (1.5679442508710802e-08 * u.s)

    nth_sidelobe_delay = get_delay_of_nth_sidelobe(sinc_width=width, n=0)
    assert nth_sidelobe_delay == (0 * u.s)
