"""Estimate how a delay notch affects spectral lines.

For a Gaussian line of FWHM ``dnu`` (in frequency) on a visibility where the
nulled object's delay is separated by ``dtau`` from the line's source, a notch
removing an effective width ``W`` of delay reduces the line peak by

    epsilon = kappa * W * dnu * exp(-(pi * dnu * dtau)**2 / (4 ln 2)),

with ``kappa = sqrt(pi / (4 ln 2))``. Lines narrower than a channel lose at most
``W * channel_width``. Over all line widths the loss is largest for
``dnu = sqrt(2 ln 2) / (pi * |dtau|)``, where it is ``0.24 * W / |dtau|``. The
notch changes the line's shape, not its integrated flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.constants import c as speed_of_light
from numpy.typing import NDArray

KAPPA = float(np.sqrt(np.pi / (4 * np.log(2))))
"""Ratio of a Gaussian's area to its FWHM times its peak"""
WORST_WIDTH_FACTOR = float(np.sqrt(2 * np.log(2)) / np.pi)
"""The line width giving the largest loss is this over the delay separation"""
WORST_LOSS_FACTOR = KAPPA * WORST_WIDTH_FACTOR * float(np.exp(-0.5))
"""The largest loss over all line widths is this times ``W / |dtau|``"""


def gaussian_line_peak_loss(
    effective_width_s: float,
    line_fwhm_hz: float | NDArray[np.floating[Any]],
    delay_separation_s: float | NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Fractional loss of a Gaussian line's peak caused by a delay notch.

    Args:
        effective_width_s (float): Effective width of the notch, in seconds
        line_fwhm_hz (float | NDArray[np.floating[Any]]): Line FWHM in frequency, in Hz
        delay_separation_s (float | NDArray[np.floating[Any]]): Delay between the nulled object and the line's source, in seconds

    Returns:
        NDArray[np.floating[Any]]: The fractional loss of the line peak
    """
    line_fwhm_hz = np.asarray(line_fwhm_hz, dtype=float)
    delay_separation_s = np.asarray(delay_separation_s, dtype=float)
    return np.asarray(
        KAPPA
        * effective_width_s
        * line_fwhm_hz
        * np.exp(-((np.pi * line_fwhm_hz * delay_separation_s) ** 2) / (4 * np.log(2)))
    )


@dataclass(frozen=True)
class LineImpactSummary:
    """The largest effect of a delay notch on spectral lines over a set of visibilities"""

    unresolved_loss: float
    """Largest fractional peak loss of a line narrower than one channel"""
    worst_loss: float
    """Largest fractional peak loss over all line widths and visibilities"""
    worst_line_fwhm_hz: float
    """Line FWHM, in Hz, at which ``worst_loss`` occurs"""
    closest_delay_s: float
    """Smallest delay separation between the object and the phase centre, in seconds"""
    fraction_above_one_percent: float
    """Fraction of visibilities in which some line width loses more than 1% of its peak"""
    reference_frequency_hz: float
    """Frequency used to quote line widths as velocities, in Hz"""
    number_of_visibilities: int
    """Number of visibilities the summary covers"""


def summarise_line_impact(
    delay_separation_s: NDArray[np.floating[Any]],
    effective_width_s: float,
    channel_width_hz: float,
    reference_frequency_hz: float,
    number_of_notches: int = 1,
) -> LineImpactSummary:
    """Summarise the largest effect of a delay notch on spectral lines.

    Args:
        delay_separation_s (NDArray[np.floating[Any]]): For each modified visibility, the delay between the nearest notch and the phase centre, in seconds
        effective_width_s (float): Effective width of one notch, in seconds
        channel_width_hz (float): The channel width, in Hz
        reference_frequency_hz (float): Frequency used to quote line widths as velocities, in Hz
        number_of_notches (int, optional): Notches per object (one, plus any nulled sidelobes). Defaults to 1.

    Returns:
        LineImpactSummary: The largest effects
    """
    separation = np.abs(np.asarray(delay_separation_s, dtype=float)).ravel()
    unresolved_loss = number_of_notches * effective_width_s * abs(channel_width_hz)
    if separation.size == 0:
        return LineImpactSummary(
            unresolved_loss=unresolved_loss,
            worst_loss=0.0,
            worst_line_fwhm_hz=np.nan,
            closest_delay_s=np.nan,
            fraction_above_one_percent=0.0,
            reference_frequency_hz=reference_frequency_hz,
            number_of_visibilities=0,
        )

    closest = float(np.min(separation))
    with np.errstate(divide="ignore"):
        worst_per_visibility = WORST_LOSS_FACTOR * effective_width_s / separation
    return LineImpactSummary(
        unresolved_loss=unresolved_loss,
        worst_loss=float(np.max(worst_per_visibility)),
        worst_line_fwhm_hz=WORST_WIDTH_FACTOR / closest if closest > 0 else np.inf,
        closest_delay_s=closest,
        fraction_above_one_percent=float(np.mean(worst_per_visibility > 0.01)),
        reference_frequency_hz=reference_frequency_hz,
        number_of_visibilities=int(separation.size),
    )


def _format_velocity(velocity_kms: float) -> str:
    """Velocities to a sensible precision, without exponents"""
    return f"{velocity_kms:,.0f}" if velocity_kms >= 10 else f"{velocity_kms:.2g}"


def describe_line_impact(object_name: str, summary: LineImpactSummary) -> str:
    """A plain description of a ``LineImpactSummary`` for logs and the HISTORY table"""
    opening = (
        f"Spectral-line impact of nulling {object_name} (Gaussian line profiles): "
        f"lines narrower than one channel lose at most "
        f"{summary.unresolved_loss * 100:.2g}% of their peak brightness."
    )
    if summary.number_of_visibilities == 0:
        return f"{opening} No visibilities were modified."

    velocity_kms = (
        summary.worst_line_fwhm_hz
        / summary.reference_frequency_hz
        * speed_of_light.to("km/s").value
    )
    return (
        f"{opening} Broader lines lose at most {summary.worst_loss * 100:.2g}% of their "
        f"peak; the largest loss is for lines about "
        f"{summary.worst_line_fwhm_hz / 1e6:.3g} MHz wide (about {_format_velocity(velocity_kms)} km/s "
        f"at {summary.reference_frequency_hz / 1e9:.3g} GHz), on the baselines where "
        f"{object_name} appears closest in delay to the phase centre "
        f"({summary.closest_delay_s * 1e9:.3g} ns). In "
        f"{summary.fraction_above_one_percent * 100:.2g}% of the modified visibilities, "
        "some line width loses more than 1% of its peak. Integrated line flux is "
        "unchanged."
    )
