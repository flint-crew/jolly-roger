"""Routines around plotting"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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

from jolly_roger.uvws import WDelays
from jolly_roger.wrap import calculate_wrapped_data, iterate_over_zones

if TYPE_CHECKING:
    from jolly_roger.delays import DelayTime
    from jolly_roger.tractor import BaselineData


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
    if w_delays is not None:
        w_delays = [w_delays] if isinstance(w_delays, WDelays) else w_delays

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

        # We may end up flagging all the data
        if not after_amp_stokesi.mask.all():
            norm = ImageNormalize(
                after_amp_stokesi
                if not after_amp_stokesi.mask.all()
                else before_amp_stokesi,
                interval=ZScaleInterval(),
                stretch=SqrtStretch(),
            )
        else:
            norm = None

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
            for _object_idx, _w_delays in enumerate(w_delays):
                plot_elevation = _w_delays.elevation.to("deg")
                ax2.plot(
                    before_baseline_data.time,
                    plot_elevation,
                    label=_w_delays.object_name,
                    color=f"C{_object_idx}",
                )
            ax2.axhline(0, lw=4, color="black", ls="-")
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
                ant_1, ant_2 = before_baseline_data.ant_1, before_baseline_data.ant_2
                b_idx = _w_delays.b_map[ant_1, ant_2]
                wrapped_data = calculate_wrapped_data(
                    values=_w_delays.w_delays[b_idx].to("ns").value,
                    upper_limit=np.max(after_delays.delay.to("ns")).value,
                )
                color_str = f"C{_object_idx}"
                for _zone_idx, object_slice in enumerate(
                    iterate_over_zones(zones=wrapped_data)
                ):
                    import matplotlib.patheffects as pe  # noqa: PLC0415

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
                        dashes=(2 * _zone_idx + 1, 2 * _zone_idx + 1),
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
        if outer_width_ns:
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
