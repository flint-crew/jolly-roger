"""Routines around plotting"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.patheffects as pe
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
from numpy.typing import NDArray

from jolly_roger.baselines import BaselineData
from jolly_roger.delays import data_to_delay_rate
from jolly_roger.logging import logger
from jolly_roger.rates import segment_spectrum
from jolly_roger.uvws import WDelays, get_w_rates
from jolly_roger.wrap import (
    axis_half_period,
    calculate_wrapped_data,
    iterate_over_zones,
    symmetric_domain_wrap,
)

if TYPE_CHECKING:
    from jolly_roger.delays import DelayTime
    from jolly_roger.rates import (
        CandidateSegment,
        SegmentFilterResult,
        SegmentGeometry,
    )


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
            after_amp_stokesi
            if not after_amp_stokesi.mask.all()
            else before_amp_stokesi
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

        # The elevation curve (ax2) has different units to ax1/3
        # so we can't share the y-axis
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            2, 3, figsize=(18, 10), sharex=True, sharey=False
        )
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
                    upper_limit=np.max(after_delays.delay.to("ns")).value,
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


def _object_extent(
    delay_ns: NDArray[np.floating[Any]],
    rate_mhz: NDArray[np.floating[Any]],
    object_delay_ns: NDArray[np.floating[Any]],
    rate_low_mhz: NDArray[np.floating[Any]],
    rate_high_mhz: NDArray[np.floating[Any]],
    width_ns: float,
) -> NDArray[np.bool_]:
    """Cells of a (delay, rate) grid an object passes through at any time,
    within ``width_ns`` of its delay. Both axes are cyclic."""
    half_delay_bin = float(np.max(np.abs(np.diff(delay_ns)))) / 2
    half_rate_bin = float(np.max(np.abs(np.diff(rate_mhz)))) / 2
    delay_offset = symmetric_domain_wrap(
        values=delay_ns[None, :] - object_delay_ns[:, None],
        upper_limit=axis_half_period(delay_ns),
    )
    within_delay = np.abs(delay_offset) <= max(width_ns, half_delay_bin)
    rate_offset = symmetric_domain_wrap(
        values=rate_mhz[None, :] - ((rate_low_mhz + rate_high_mhz) / 2)[:, None],
        upper_limit=axis_half_period(rate_mhz),
    )
    rate_half_width = np.maximum(
        np.abs(rate_high_mhz - rate_low_mhz) / 2, half_rate_bin
    )
    within_rate = np.abs(rate_offset) <= rate_half_width[:, None]
    return (within_delay.T.astype(float) @ within_rate.astype(float)) > 0


def _plot_object_path(
    ax: plt.Axes,
    rate_mhz: NDArray[np.floating[Any]],
    delay_ns: NDArray[np.floating[Any]],
    rate_axis_mhz: NDArray[np.floating[Any]],
    delay_axis_ns: NDArray[np.floating[Any]],
    color: str,
    label: str | None,
) -> None:
    """An object's path in fringe rate (x) and delay (y), broken where it wraps"""
    wrapped_delay = calculate_wrapped_data(
        values=delay_ns, upper_limit=axis_half_period(delay_axis_ns)
    )
    wrapped_rate = calculate_wrapped_data(
        values=rate_mhz, upper_limit=axis_half_period(rate_axis_mhz)
    )
    zones = wrapped_delay.zones * (np.max(wrapped_rate.zones) + 1) + np.asarray(
        wrapped_rate.zones
    )
    for index, piece in enumerate(iterate_over_zones(zones=zones)):
        ax.plot(
            wrapped_rate.values[piece],
            wrapped_delay.values[piece],
            color=color,
            lw=3,
            path_effects=[pe.Stroke(linewidth=4, foreground="k"), pe.Normal()],
            label=label if index == 0 else None,
        )


def plot_baseline_delay_rate_comparison(
    before_baseline_data: BaselineData,
    after_baseline_data: BaselineData,
    output_path: Path,
    w_delays: list[WDelays] | None = None,
    outer_width_ns: float | None = None,
) -> Path:
    """A baseline before and after nulling, in fringe rate and delay over the
    whole observation, with the path of each object and the field's guard region

    Args:
        before_baseline_data (BaselineData): The baseline before nulling
        after_baseline_data (BaselineData): The baseline after nulling
        output_path (Path): Where to save the figure
        w_delays (list[WDelays] | None, optional): The objects nulled. Defaults to None.
        outer_width_ns (float | None, optional): The notch's outer width. Defaults to None.

    Returns:
        Path: The saved figure
    """
    before = data_to_delay_rate(before_baseline_data)
    after = data_to_delay_rate(after_baseline_data)
    delay_ns = before.delay.to("ns").value
    rate_mhz = before.rate.to("mHz").value
    freq_hz = before_baseline_data.freq_chan.to("Hz").value
    nu_low, nu_high = float(freq_hz.min()), float(freq_hz.max())

    fig, (ax_before, ax_map, ax_after) = plt.subplots(1, 3, figsize=(18, 5))
    amplitudes = [
        np.abs((delay_rate.delay_rate[..., 0] + delay_rate.delay_rate[..., -1]) / 2)
        for delay_rate in (before, after)
    ]
    norm = ImageNormalize(
        amplitudes[0], interval=MinMaxInterval(), stretch=LogStretch()
    )
    for ax, amplitude, title in zip(
        (ax_before, ax_after), amplitudes, ("Before", "After"), strict=True
    ):
        im = ax.pcolormesh(
            rate_mhz, delay_ns, amplitude.T, norm=norm, shading="nearest"
        )
        ax.set(xlabel="Fringe rate / mHz", ylabel="Delay / ns", title=title)
        fig.colorbar(im, ax=ax, label="Stokes I Amplitude / Jy")

    width_ns = outer_width_ns or 0.0
    delay_guard_ns, rate_guard_mhz = width_ns, 0.0
    if w_delays:
        b_idx = w_delays[0].b_map[
            before_baseline_data.ant_1, before_baseline_data.ant_2
        ]
        for index, object_delays in enumerate(w_delays):
            color = f"C{index}"
            object_delay_ns = object_delays.w_delays[b_idx].to("ns").value
            rate = get_w_rates(object_delays)[b_idx].to("").value
            extent = _object_extent(
                delay_ns=delay_ns,
                rate_mhz=rate_mhz,
                object_delay_ns=object_delay_ns,
                rate_low_mhz=rate * nu_low * 1e3,
                rate_high_mhz=rate * nu_high * 1e3,
                width_ns=width_ns,
            )
            if extent.any():
                ax_map.contourf(
                    rate_mhz,
                    delay_ns,
                    extent.astype(float),
                    levels=[0.5, 1.5],
                    colors=[color],
                    alpha=0.25,
                )
            _plot_object_path(
                ax=ax_map,
                rate_mhz=rate * (nu_low + nu_high) / 2 * 1e3,
                delay_ns=object_delay_ns,
                rate_axis_mhz=rate_mhz,
                delay_axis_ns=delay_ns,
                color=color,
                label=object_delays.object_name,
            )
        first = w_delays[0]
        if first.guard_region is not None:
            delay_guard_ns += float(np.max(first.guard_region[b_idx].to("ns").value))
        if first.rate_guard_region is not None:
            rate_guard_mhz = (
                nu_high
                * float(np.max(first.rate_guard_region[b_idx].to("").value))
                * 1e3
            )

    ax_map.plot(0, 0, marker="o", color="black", ls="none", label="Field")
    ax_map.add_patch(
        Rectangle(
            (-rate_guard_mhz, -delay_guard_ns),
            2 * rate_guard_mhz,
            2 * delay_guard_ns,
            alpha=0.3,
            color="grey",
            label="Guard region",
        )
    )
    ax_map.legend(loc="upper right")
    ax_map.grid()
    ax_map.set(
        xlim=(rate_mhz.min(), rate_mhz.max()),
        ylim=(delay_ns.min(), delay_ns.max()),
        xlabel="Fringe rate / mHz",
        ylabel="Delay / ns",
        title="Objects and field",
    )
    fig.suptitle(
        f"Ant {after_baseline_data.ant_1} - Ant {after_baseline_data.ant_2} (delay-rate)"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_rate_filter_segment(
    segment: CandidateSegment,
    result: SegmentFilterResult,
    geometry: SegmentGeometry,
    freq_hz: NDArray[np.floating[Any]],
    output_path: Path,
) -> Path:
    """A segment before and after delay-rate filtering, in fringe rate and
    delay, with each object's path, the region removed and the field's box

    Args:
        segment (CandidateSegment): The rows filtered
        result (SegmentFilterResult): The result of ``filter_segment``
        geometry (SegmentGeometry): The objects' and field's delays and rates
        freq_hz (NDArray[np.floating[Any]]): Frequency of each channel, in Hz
        output_path (Path): Where to save the figure

    Returns:
        Path: The saved figure
    """
    assert result.data is not None
    assert result.removal is not None
    assert result.window is not None
    flags = np.array(segment.mask)
    n_time = result.removal.shape[0]
    step_s = float(np.median(np.diff(segment.time_mjds)))
    rate_mhz = np.fft.fftshift(np.fft.fftfreq(n_time, step_s)) * 1e3
    delay_ns = (
        np.fft.fftshift(np.fft.fftfreq(len(freq_hz), np.mean(np.diff(freq_hz)))) * 1e9
    )
    before = segment_spectrum(np.array(segment.data), flags, result.window, n_time)
    amplitudes = [
        np.fft.fftshift(before),
        np.fft.fftshift(before * (1 - result.removal)),
    ]
    positive = amplitudes[0][amplitudes[0] > 0]
    norm = SymLogNorm(
        linthresh=float(np.percentile(positive, 5)) if positive.size else 1e-6,
        vmin=0.0,
        vmax=float(positive.max()) if positive.size else 1.0,
    )
    nu_low, nu_high = float(np.min(freq_hz)), float(np.max(freq_hz))
    field_delay_ns = float(np.max(geometry.field_delay_s)) * 1e9
    field_rate_mhz = (
        float(np.max(geometry.field_rate)) * nu_high + 1 / (len(segment.rows) * step_s)
    ) * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, amplitude, title in zip(axes, amplitudes, ("Before", "After"), strict=True):
        im = ax.pcolormesh(
            rate_mhz, delay_ns, amplitude.T, norm=norm, shading="nearest"
        )
        ax.add_patch(
            Rectangle(
                (-field_rate_mhz, -field_delay_ns),
                2 * field_rate_mhz,
                2 * field_delay_ns,
                fill=False,
                edgecolor="white",
                linewidth=1.5,
                label="Field",
            )
        )
        for index, name in enumerate(geometry.object_names):
            active = geometry.object_active[index]
            if not active.any():
                continue
            _plot_object_path(
                ax=ax,
                rate_mhz=geometry.object_rate[index][active]
                * (nu_low + nu_high)
                / 2
                * 1e3,
                delay_ns=geometry.object_delay_s[index][active] * 1e9,
                rate_axis_mhz=rate_mhz,
                delay_axis_ns=delay_ns,
                color=f"C{index + 1}",
                label=name,
            )
        removal = np.fft.fftshift(result.removal)
        if (removal > 0.5).any():
            ax.contour(
                rate_mhz,
                delay_ns,
                removal.T,
                levels=[0.5],
                colors=["C1"],
                linestyles="--",
                linewidths=1.5,
            )
            ax.plot([], [], color="C1", ls="--", lw=1.5, label="Removed")
        ax.set(xlabel="Fringe rate / mHz", title=title)
    axes[0].set_ylabel("Delay / ns")
    axes[0].legend(loc="upper right", fontsize="small")
    fig.colorbar(im, ax=axes, label="Stokes I Amplitude / Jy")
    recovered = 0 if result.recovered is None else int(result.recovered.sum())
    fig.suptitle(
        f"Ant {segment.ant_1} - Ant {segment.ant_2}, rows {segment.rows[0]}-{segment.rows[-1]}: "
        f"{recovered} of {segment.n_candidates} candidates recovered"
    )
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
