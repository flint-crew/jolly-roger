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
from jolly_roger.wrap import calculate_nyquist_zone, symmetric_domain_wrap

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
    w_delays: WDelays | None = None,
    outer_width_ns: float | None = None,
) -> Path:
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

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            2, 3, figsize=(18, 10), sharex=True, sharey="row"
        )
        im = ax1.pcolormesh(
            before_baseline_data.time,
            before_baseline_data.freq_chan,
            before_amp_stokesi.T,
            norm=norm,
            cmap=cmap,
        )
        ax2.set_axis_off()
        ax1.set(
            ylabel=f"Frequency / {before_baseline_data.freq_chan.unit:latex_inline}",
            title="Before",
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
            ant_1, ant_2 = before_baseline_data.ant_1, before_baseline_data.ant_2
            b_idx = w_delays.b_map[ant_1, ant_2]
            wrapped_w_delays = symmetric_domain_wrap(
                values=w_delays.w_delays[b_idx].to("ns").value,
                upper_limit=np.max(after_delays.delay.to("ns")).value,
            )
            zones = calculate_nyquist_zone(
                values=w_delays.w_delays[b_idx].to("ns").value,
                upper_limit=np.max(after_delays.delay.to("ns")).value,
            )
            # Final append is to capture the last zone in the
            # time range
            transitions = [*np.argwhere(np.diff(zones) != 0)[:, 0], len(zones)]
            start_idx = 0
            for _zone_idx, end_idx in enumerate(transitions):
                # The np.diff results in offset indices, so shift
                # back the transition by 1
                object_slice = slice(start_idx, end_idx + 1)
                # and ensure non-overlapping line segments
                start_idx = end_idx + 1
                import matplotlib.patheffects as pe  # noqa: PLC0415

                ax5.plot(
                    before_baseline_data.time[object_slice],
                    wrapped_w_delays[object_slice],
                    color="tab:red",
                    label=f"Delay for {w_delays.object_name}"
                    if _zone_idx == 0
                    else None,
                    lw=5,
                    path_effects=[
                        pe.Stroke(
                            linewidth=6, foreground="k"
                        ),  # Add some contrast to help read line stand out
                        pe.Normal(),
                    ],
                    dashes=(2 * _zone_idx + 1, 2 * _zone_idx + 1),
                )

            if outer_width_ns is not None:
                for s, sign in enumerate((1, -1)):
                    wrapped_outer_width = symmetric_domain_wrap(
                        values=wrapped_w_delays + outer_width_ns * sign,
                        upper_limit=np.max(after_delays.delay.to("ns")).value,
                    )
                    outer_zones = calculate_nyquist_zone(
                        values=wrapped_w_delays + outer_width_ns * sign,
                        upper_limit=np.max(after_delays.delay.to("ns")).value,
                    )
                    transitions = [
                        *np.argwhere(np.diff(outer_zones) != 0)[:, 0],
                        len(zones),
                    ]
                    start_idx = 0
                    for _zone_idx, end_idx in enumerate(transitions):
                        object_slice = slice(start_idx, end_idx + 1)
                        start_idx = end_idx + 1
                        ax5.plot(
                            before_baseline_data.time[object_slice],
                            wrapped_outer_width[object_slice],
                            ls="--",
                            color="k",
                            lw=1,
                            label="outer_width" if _zone_idx == 0 and s == 0 else None,
                        )

                ax5.axhline(0, ls="-", c="black", label="Field", lw=4)
                ax5.axhspan(
                    -outer_width_ns,
                    outer_width_ns,
                    alpha=0.3,
                    color="grey",
                    label="Contamination",
                )

        ax5.legend(loc="upper right")
        ax5.grid()

        fig.suptitle(
            f"Ant {after_baseline_data.ant_1} - Ant {after_baseline_data.ant_2}"
        )
        fig.tight_layout()
        fig.savefig(output_path)

        return output_path
