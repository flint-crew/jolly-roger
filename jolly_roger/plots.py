"""Routines around plotting"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from jolly_roger.uvws import WDelays

if TYPE_CHECKING:
    from jolly_roger.delays import DelayTime
    from jolly_roger.tractor import BaselineData


def plot_baseline_data(
    baseline_data: BaselineData,
    output_dir: Path,
    suffix: str = "",
) -> None:
    from astropy.visualization import quantity_support, time_support

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
    output_dir: Path,
    suffix: str = "",
    w_delays: WDelays | None = None,
) -> Path:
    from astropy.visualization import (
        ImageNormalize,
        LogStretch,
        MinMaxInterval,
        SqrtStretch,
        ZScaleInterval,
        quantity_support,
        time_support,
    )

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

        norm = ImageNormalize(
            after_amp_stokesi, interval=ZScaleInterval(), stretch=SqrtStretch()
        )

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        im = ax1.pcolormesh(
            before_baseline_data.time,
            before_baseline_data.freq_chan,
            before_amp_stokesi.T,
            norm=norm,
        )
        ax1.set(
            ylabel=f"Frequency / {before_baseline_data.freq_chan.unit:latex_inline}",
            title="Before",
        )
        ax2.pcolormesh(
            after_baseline_data.time,
            after_baseline_data.freq_chan,
            after_amp_stokesi.T,
            norm=norm,
        )
        ax2.set(
            ylabel=f"Frequency / {after_baseline_data.freq_chan.unit:latex_inline}",
            title="After",
        )
        fig.colorbar(im, ax=ax2, label="Stokes I Amplitude / Jy")

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

        im = ax3.pcolormesh(
            before_baseline_data.time,
            before_delays.delay,
            before_delays_i.T,
            norm=delay_norm,
        )
        ax3.set(ylabel="Delay / s", title="Before")
        ax4.pcolormesh(
            after_baseline_data.time,
            after_delays.delay,
            after_delays_i.T,
            norm=delay_norm,
        )
        ax4.set(ylabel="Delay / s", title="After")
        fig.colorbar(im, ax=ax4, label="Stokes I Amplitude / Jy")

        if w_delays is not None:
            for ax in (ax3, ax4):
                ax.plot(
                    before_delays.time,
                    w_delays.w_delay,
                    color="k",
                    linestyle="--",
                    label=f"Delay for {w_delays.object_name}",
                )
                ax.legend()

        output_path = (
            output_dir
            / f"baseline_data_{before_baseline_data.ant_1}_{before_baseline_data.ant_2}{suffix}.png"
        )
        fig.suptitle(
            f"Ant {after_baseline_data.ant_1} - Ant {after_baseline_data.ant_2}"
        )
        fig.tight_layout()
        fig.savefig(output_path)

        return output_path
