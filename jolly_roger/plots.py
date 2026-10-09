"""Routines around plotting"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import (
    ImageNormalize,
    LogStretch,
    MinMaxInterval,
    SqrtStretch,
    ZScaleInterval,
    quantity_support,
    time_support,
)
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle

from jolly_roger.baselines import BaselineData
from jolly_roger.logging import logger
from jolly_roger.uvws import WDelays, get_w_rates
from jolly_roger.wrap import calculate_wrapped_data, iterate_over_zones

if TYPE_CHECKING:
    from jolly_roger.delays import DelayRate, DelayTime
    from jolly_roger.rates import RateBox, RateFilterDiagnostics


def plot_baseline_data(
    baseline_data: BaselineData,
    output_dir: Path,
    suffix: str = "",
) -> None:
    with quantity_support(), time_support():
        data_masked = baseline_data.masked_data
        data_xx = data_masked[..., 0]
        data_yy = data_masked[..., -1]
        data_stokesi = (data_xx + data_yy) / 2
        amp_stokesi = np.abs(data_stokesi)

        fig, ax = plt.subplots()
        im = ax.pcolormesh(
            baseline_data.time,
            baseline_data.freq_chan,
            amp_stokesi.T,
        )
        fig.colorbar(im, ax=ax, label="Stokes I Amplitude / Jy")
        ax.set(
            ylabel=f"Frequency / {baseline_data.freq_chan.unit:latex_inline}",
            title=f"Ant {baseline_data.ant_1} - Ant {baseline_data.ant_2}",
        )
        output_path = (
            output_dir
            / f"baseline_data_{baseline_data.ant_1}_{baseline_data.ant_2}{suffix}.png"
        )
        fig.savefig(output_path)


def _plot_dynamic_spectra_row(
    fig: plt.Figure,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    before_baseline_data: BaselineData,
    after_baseline_data: BaselineData,
    w_delays: list[WDelays] | None,
    b_idx: int | None,
    max_delay_ns: float,
) -> None:
    """Draw the before and after dynamic spectra (time vs frequency) and, between
    them, the elevation and Nyquist zone of each object. Must be called within
    ``quantity_support`` and ``time_support``.

    Args:
        fig (plt.Figure): The figure the axes belong to
        axes (tuple[plt.Axes, plt.Axes, plt.Axes]): The before, object and after axes
        before_baseline_data (BaselineData): The baseline data from the before state
        after_baseline_data (BaselineData): The baseline data from the after state
        w_delays (list[WDelays] | None): Delays corresponding to objects that have been nulled
        b_idx (int | None): The index of the baseline into the ``w_delays``
        max_delay_ns (float): The largest delay of the delay spectrum, in ns, used to compute Nyquist zones
    """
    ax1, ax2, ax3 = axes
    before_amp_stokesi = np.abs(
        (
            before_baseline_data.masked_data[..., 0]
            + before_baseline_data.masked_data[..., -1]
        )
        / 2
    )
    after_amp_stokesi = np.abs(
        (
            after_baseline_data.masked_data[..., 0]
            + after_baseline_data.masked_data[..., -1]
        )
        / 2
    )

    # We may end up flagging all the data. If the after data is completely flagged, fall back
    # to the before data. If, however, all that is also flagged (e.g. sun is too close across
    # all timesteps and hence all timesteps are flagged) we no normalise.
    norm = None
    norm_plot_data = (
        after_amp_stokesi if not after_amp_stokesi.mask.all() else before_amp_stokesi
    )
    if not norm_plot_data.mask.all():
        norm = ImageNormalize(
            norm_plot_data,
            interval=ZScaleInterval(),
            stretch=SqrtStretch(),
        )

    else:
        logger.warning("No valid data found. No attempt to normalise data.")

    cmap = plt.cm.viridis

    im = ax1.pcolormesh(
        before_baseline_data.time,
        before_baseline_data.freq_chan,
        before_amp_stokesi.T,
        norm=norm,
        cmap=cmap,
    )
    ax1.set(
        ylabel=f"Frequency / {before_baseline_data.freq_chan.unit:latex_inline}",
        title="Before",
    )

    ax2.set_axis_off()
    if w_delays:
        assert b_idx is not None, "A baseline index is needed to plot objects"
        ax2.set_axis_on()
        ax2_zone = ax2.twinx()
        ax2.axhline(0, lw=4, color="black", ls="-")

        max_zone = 0
        for _object_idx, _w_delays in enumerate(w_delays):
            plot_elevation = _w_delays.elevation.to("deg")
            ax2.plot(
                before_baseline_data.time,
                plot_elevation,
                label=_w_delays.object_name,
                color=f"C{_object_idx}",
            )
            plot_delay = _w_delays.w_delays[b_idx].to("ns").value
            plot_zone = calculate_wrapped_data(
                values=plot_delay,
                upper_limit=max_delay_ns,
            )
            ax2_zone.plot(before_baseline_data.time, plot_zone.zones, ls="--")
            object_max_zone = max(plot_zone.zones)
            max_zone = object_max_zone if object_max_zone > max_zone else max_zone
        ax2_zone.set(ylabel="Nyquist Zone", ylim=[0, max_zone + 1])
        ax2.legend()
        ax2.grid()
        ax2.set(
            ylabel=f"Elevation / {plot_elevation.unit:latex_inline}",
            ylim=[-90.0, 90.0],
        )

    ax3.pcolormesh(
        after_baseline_data.time,
        after_baseline_data.freq_chan,
        after_amp_stokesi.T,
        norm=norm,
        cmap=cmap,
    )
    ax3.set(
        ylabel=f"Frequency / {after_baseline_data.freq_chan.unit:latex_inline}",
        title="After",
    )
    for ax in (ax1, ax3):
        fig.colorbar(im, ax=ax, label="Stokes I Amplitude / Jy")


def plot_baseline_comparison_data(
    before_baseline_data: BaselineData,
    after_baseline_data: BaselineData,
    before_delays: DelayTime,
    after_delays: DelayTime,
    output_path: Path,
    w_delays: WDelays | list[WDelays] | None = None,
    outer_width_ns: float | None = None,
) -> Path:
    """Make a comparison figure showing the before and after the nulling process,
    including an examination spectral axis and the formed delay spectrum.
    Each object being nulled also is included as an overlap to highlight the oath
    that is taken through the delay space.

    Args:
        before_baseline_data (BaselineData): The baseline data from the before state
        after_baseline_data (BaselineData): The baseline data from the after state
        before_delays (DelayTime): Delays formed that correspond to the spectral axis from the before state
        after_delays (DelayTime): Delays formed that correspond to the spectral axis from the after state
        output_path (Path): The location that the figure will be saved to
        w_delays (WDelays | list[WDelays] | None, optional): Delays corresponding to objects that have been nulled. Defaults to None.
        outer_width_ns (float | None, optional): The taper size. Defaults to None.

    Returns:
        Path: Path to the saved figure
    """
    if w_delays is not None:
        w_delays = [w_delays] if isinstance(w_delays, WDelays) else w_delays

        # By construction all the WDelay objects will be from the same array
        ant_1, ant_2 = before_baseline_data.ant_1, before_baseline_data.ant_2
        b_idx = w_delays[0].b_map[ant_1, ant_2]

    with quantity_support(), time_support():
        cmap = plt.cm.viridis

        # The elevation curve (ax2) has different units to ax1/3
        # so we can't share the y-axis
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            2, 3, figsize=(18, 10), sharex=True, sharey=False
        )
        _plot_dynamic_spectra_row(
            fig=fig,
            axes=(ax1, ax2, ax3),
            before_baseline_data=before_baseline_data,
            after_baseline_data=after_baseline_data,
            w_delays=w_delays,
            b_idx=b_idx if w_delays is not None else None,
            max_delay_ns=float(np.max(after_delays.delay.to("ns")).value),
        )

        # TODO: Move these delay calculations outside of the plotting function
        # And here we calculate the delay information

        before_delays_i = np.abs(
            (before_delays.delay_time[:, :, 0] + before_delays.delay_time[:, :, -1]) / 2
        )
        after_delays_i = np.abs(
            (after_delays.delay_time[:, :, 0] + after_delays.delay_time[:, :, -1]) / 2
        )

        delay_norm = ImageNormalize(
            before_delays_i, interval=MinMaxInterval(), stretch=LogStretch()
        )

        im = ax4.pcolormesh(
            before_baseline_data.time,
            before_delays.delay.to("ns"),
            before_delays_i.T,
            norm=delay_norm,
            cmap=cmap,
        )
        ax4.set(ylabel="Delay / ns", title="Before")
        ax6.pcolormesh(
            after_baseline_data.time,
            after_delays.delay.to("ns"),
            after_delays_i.T,
            norm=delay_norm,
            cmap=cmap,
        )
        ax6.set(ylabel="Delay / ns", title="After")
        for ax in (ax4, ax6):
            fig.colorbar(im, ax=ax, label="Stokes I Amplitude / Jy")

        if w_delays is not None:
            for _object_idx, _w_delays in enumerate(w_delays):
                wrapped_data = calculate_wrapped_data(
                    values=_w_delays.w_delays[b_idx].to("ns").value,
                    upper_limit=np.max(after_delays.delay.to("ns")).value,
                )
                color_str = f"C{_object_idx}"
                for _zone_idx, object_slice in enumerate(
                    iterate_over_zones(zones=wrapped_data)
                ):
                    import matplotlib.patheffects as pe  # noqa: PLC0415

                    current_zone = np.mean(wrapped_data.zones[object_slice])
                    ax5.plot(
                        before_baseline_data.time[object_slice],
                        wrapped_data.values[object_slice],
                        color=color_str,
                        label=f"Delay for {_w_delays.object_name}"
                        if _zone_idx == 0
                        else None,
                        lw=3,
                        path_effects=[
                            pe.Stroke(
                                linewidth=4, foreground="k"
                            ),  # Add some contrast to help read line stand out
                            pe.Normal(),
                        ],
                        dashes=(1.2 * current_zone + 1, 1.2 * current_zone + 1),
                    )

                if outer_width_ns is not None:
                    for s, sign in enumerate((1, -1)):
                        wrapped_outer_data = calculate_wrapped_data(
                            values=wrapped_data.values + outer_width_ns * sign,
                            upper_limit=np.max(after_delays.delay.to("ns")).value,
                        )
                        # for _zone_idx, end_idx in enumerate(transitions):
                        for _zone_idx, object_slice in enumerate(
                            iterate_over_zones(zones=wrapped_outer_data)
                        ):
                            ax5.plot(
                                before_baseline_data.time[object_slice],
                                wrapped_outer_data.values[object_slice],
                                ls=":",
                                color=color_str,
                                lw=2,
                                label="outer_width"
                                if _zone_idx == 0 and s == 0 and _object_idx == 0
                                else None,
                            )

        ax5.axhline(0, ls="-", c="black", label="Field", lw=4)
        if w_delays is not None and w_delays[0].guard_region is not None:
            baseline_guard = w_delays[0].guard_region[b_idx].to("ns").value
            if outer_width_ns:
                baseline_guard += outer_width_ns
            logger.info(
                f"Adding to baseline guard, maximum {np.max(baseline_guard):.3f}"
            )
            ax5.fill_between(
                before_baseline_data.time,
                -baseline_guard,
                baseline_guard,
                alpha=0.3,
                color="grey",
                label="Guard Region",
            )

        elif outer_width_ns:
            ax5.axhspan(
                -outer_width_ns,
                outer_width_ns,
                alpha=0.3,
                color="grey",
                label="Contamination",
            )

        ax5.legend(loc="upper right")
        ax5.grid()
        ax5.set(
            ylim=[
                np.min(after_delays.delay.to("ns")),
                np.max(after_delays.delay.to("ns")),
            ],
            ylabel="Delay / ns",
        )

        fig.suptitle(
            f"Ant {after_baseline_data.ant_1} - Ant {after_baseline_data.ant_2}"
        )
        fig.tight_layout()
        fig.savefig(output_path)

        return output_path


def plot_baseline_delay_rate_comparison(
    before_baseline_data: BaselineData,
    after_baseline_data: BaselineData,
    before_delay_rate: DelayRate,
    after_delay_rate: DelayRate,
    output_path: Path,
    w_delays: WDelays | list[WDelays] | None = None,
    outer_width_ns: float | None = None,
) -> Path:
    """Make a comparison figure of a baseline before and after the nulling process,
    in delay and delay-rate across the whole observation. The top row matches
    ``plot_baseline_comparison_data``. The bottom row replaces delay vs time with
    delay vs delay-rate, where an object moving through delay separates from the
    field. The path of each object, and the region occupied by the field, are
    shown between the before and after panels.

    Args:
        before_baseline_data (BaselineData): The baseline data from the before state
        after_baseline_data (BaselineData): The baseline data from the after state
        before_delay_rate (DelayRate): The delay-rate transform of the before state
        after_delay_rate (DelayRate): The delay-rate transform of the after state
        output_path (Path): The location that the figure will be saved to
        w_delays (WDelays | list[WDelays] | None, optional): Delays corresponding to objects that have been nulled. Defaults to None.
        outer_width_ns (float | None, optional): The taper size. Defaults to None.

    Returns:
        Path: Path to the saved figure
    """
    b_idx: int | None = None
    if w_delays is not None:
        w_delays = [w_delays] if isinstance(w_delays, WDelays) else w_delays
        b_idx = w_delays[0].b_map[
            before_baseline_data.ant_1, before_baseline_data.ant_2
        ]

    delay_ns = before_delay_rate.delay.to("ns").value
    rate_mhz = before_delay_rate.rate.to("mHz").value
    freq_hz = before_baseline_data.freq_chan.to("Hz").value
    nu_mid, nu_max = (
        float(np.mean([freq_hz.min(), freq_hz.max()])),
        float(freq_hz.max()),
    )

    with quantity_support(), time_support():
        cmap = plt.cm.viridis
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            2, 3, figsize=(18, 10), sharex=False, sharey=False
        )
        # Only the top row shares a (time) axis
        ax2.sharex(ax1)
        ax3.sharex(ax1)
        _plot_dynamic_spectra_row(
            fig=fig,
            axes=(ax1, ax2, ax3),
            before_baseline_data=before_baseline_data,
            after_baseline_data=after_baseline_data,
            w_delays=w_delays,
            b_idx=b_idx,
            max_delay_ns=float(np.max(delay_ns)),
        )

    before_rate_i = np.abs(
        (before_delay_rate.delay_rate[..., 0] + before_delay_rate.delay_rate[..., -1])
        / 2
    )
    after_rate_i = np.abs(
        (after_delay_rate.delay_rate[..., 0] + after_delay_rate.delay_rate[..., -1]) / 2
    )
    rate_norm = ImageNormalize(
        before_rate_i, interval=MinMaxInterval(), stretch=LogStretch()
    )
    for ax, amplitude, title in (
        (ax4, before_rate_i, "Before"),
        (ax6, after_rate_i, "After"),
    ):
        # Delay is along the y-axis, as in the delay vs time panels
        im = ax.pcolormesh(
            rate_mhz,
            delay_ns,
            amplitude.T,
            norm=rate_norm,
            cmap=cmap,
            shading="nearest",
        )
        ax.set(xlabel="Fringe-rate / mHz", ylabel="Delay / ns", title=title)
        fig.colorbar(im, ax=ax, label="Stokes I Amplitude / Jy")

    # The path each object takes through delay and delay-rate, at the central frequency
    if w_delays is not None and b_idx is not None:
        import matplotlib.patheffects as pe  # noqa: PLC0415

        for _object_idx, _w_delays in enumerate(w_delays):
            wrapped_data = calculate_wrapped_data(
                values=_w_delays.w_delays[b_idx].to("ns").value,
                upper_limit=float(np.max(delay_ns)),
            )
            object_rate_mhz = nu_mid * get_w_rates(_w_delays)[b_idx].value * 1e3
            for _zone_idx, object_slice in enumerate(
                iterate_over_zones(zones=wrapped_data)
            ):
                current_zone = np.mean(wrapped_data.zones[object_slice])
                ax5.plot(
                    object_rate_mhz[object_slice],
                    wrapped_data.values[object_slice],
                    color=f"C{_object_idx}",
                    label=f"Path of {_w_delays.object_name}"
                    if _zone_idx == 0
                    else None,
                    lw=3,
                    path_effects=[
                        pe.Stroke(linewidth=4, foreground="k"),
                        pe.Normal(),
                    ],
                    dashes=(1.2 * current_zone + 1, 1.2 * current_zone + 1),
                )

    # The region occupied by the field
    delay_guard_ns = outer_width_ns or 0.0
    rate_guard_mhz = 0.0
    if w_delays is not None and b_idx is not None:
        if w_delays[0].guard_region is not None:
            delay_guard_ns += float(
                np.max(w_delays[0].guard_region[b_idx].to("ns").value)
            )
        if w_delays[0].rate_guard_region is not None:
            rate_guard_mhz = (
                nu_max * float(np.max(w_delays[0].rate_guard_region[b_idx].value)) * 1e3
            )
    ax5.plot(0, 0, marker="o", color="black", ls="none", label="Field")
    if rate_guard_mhz > 0:
        ax5.add_patch(
            Rectangle(
                (-rate_guard_mhz, -delay_guard_ns),
                2 * rate_guard_mhz,
                2 * delay_guard_ns,
                alpha=0.3,
                color="grey",
                label="Guard Region",
            )
        )
    else:
        # Without a guard in delay-rate the field region is only extended in delay
        ax5.axvline(0, ls="-", c="black", lw=1)
        if delay_guard_ns > 0:
            ax5.plot(
                [0, 0],
                [-delay_guard_ns, delay_guard_ns],
                lw=8,
                alpha=0.3,
                color="grey",
                solid_capstyle="butt",
                label="Guard Region",
            )

    ax5.legend(loc="upper right")
    ax5.grid()
    ax5.set(
        xlim=[np.min(rate_mhz), np.max(rate_mhz)],
        ylim=[np.min(delay_ns), np.max(delay_ns)],
        xlabel="Fringe-rate / mHz",
        ylabel="Delay / ns",
    )

    fig.suptitle(
        f"Ant {after_baseline_data.ant_1} - Ant {after_baseline_data.ant_2} (delay-rate)"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def _add_rate_box(ax: plt.Axes, box: RateBox, **kwargs: Any) -> None:
    """Draw a delay (ns) and fringe-rate (mHz) box. Wrapping is not drawn."""
    ax.add_patch(
        Rectangle(
            (
                (box.delay_center_s - box.delay_half_width_s) * 1e9,
                (box.rate_center_hz - box.rate_half_width_hz) * 1e3,
            ),
            2 * box.delay_half_width_s * 1e9,
            2 * box.rate_half_width_hz * 1e3,
            fill=False,
            **kwargs,
        )
    )


def plot_rate_filter_segment(
    diagnostics: RateFilterDiagnostics,
    output_path: Path,
) -> Path:
    """Plot the delay vs delay-rate amplitude of a segment before and after
    delay-rate filtering, with the predicted track of each object, the region
    nulled for each object, and the protected field region overlaid.

    Args:
        diagnostics (RateFilterDiagnostics): Description of the filtered segment
        output_path (Path): The location that the figure will be saved to

    Returns:
        Path: The location of the saved figure
    """
    delay_ns = diagnostics.delay_s * 1e9
    rate_mhz = diagnostics.rate_hz * 1e3

    # Shared colour scale so the before and after are comparable. The scale is
    # linear below ``linthresh`` so values of zero (e.g. where the taper nulls the
    # object) and the taper's roll-off are drawn, rather than masked as on a log scale
    positive = diagnostics.before[diagnostics.before > 0]
    linthresh, vmax = (
        (np.percentile(positive, 5), np.max(positive)) if positive.size else (1e-6, 1.0)
    )
    norm = SymLogNorm(linthresh=linthresh, vmin=0.0, vmax=vmax, base=10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, amplitude, title in zip(
        axes, (diagnostics.before, diagnostics.after), ("Before", "After"), strict=True
    ):
        im = ax.pcolormesh(
            delay_ns,
            rate_mhz,
            amplitude,
            norm=norm,
            shading="nearest",
        )
        _add_rate_box(
            ax, diagnostics.field, edgecolor="white", linewidth=1.5, label="Field"
        )
        for object_idx, track in enumerate(diagnostics.tracks):
            color = f"C{object_idx + 1}"
            ax.plot(
                track.delay_s * 1e9,
                track.rate_hz * 1e3,
                color=color,
                marker=".",
                markersize=3,
                linewidth=1,
                label=track.object_name,
            )
            _add_rate_box(
                ax, track.notch, edgecolor=color, linestyle="--", linewidth=1.5
            )
        ax.set(
            xlabel="Delay / ns",
            title=title,
            xlim=(delay_ns.min(), delay_ns.max()),
            ylim=(rate_mhz.min(), rate_mhz.max()),
        )
    axes[0].set_ylabel("Fringe-rate / mHz")
    axes[0].legend(loc="upper right", fontsize="small")
    fig.colorbar(im, ax=axes, label="Amplitude")
    fig.suptitle(
        f"Ant {diagnostics.ant_1} - Ant {diagnostics.ant_2}, segment from row {diagnostics.first_row}"
    )

    fig.savefig(output_path)
    plt.close(fig)
    logger.debug(f"Saved {output_path=}")

    return output_path
