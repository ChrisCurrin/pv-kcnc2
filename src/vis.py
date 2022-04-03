import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.constants import (
    AIS_LABEL,
    DISTANCE_LABEL,
    SITE_LABEL,
    TERMINAL_LABEL,
    TIME_LABEL,
    VOLTAGE_LABEL,
)
from src.data import concise_df, get_file_path, is_long_form, wide_to_long
from src.settings import SECTION_PALETTE, STIM_ONSET, STIM_PULSE_DUR

logger = logging.getLogger("vis")


def set_default_style():
    sns.set_theme(
        context="paper",  # poster or paper
        style="ticks",
        palette="colorblind",
        rc={
            "pdf.fonttype": 42,  # embed font in output
            "svg.fonttype": "none",  # embed font in output
            "figure.facecolor": "white",
            "figure.dpi": 200,
            "legend.frameon": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "savefig.bbox": "tight",
        },
    )


def plot_voltage_trace(
    df: pd.DataFrame,
    concise=False,
    thresh=-20,
    offset=False,
    edge=False,
    ax_props: dict = None,
    ax=None,
    **kwargs,
):
    legend = kwargs.pop("legend", "brief")
    palette = kwargs.pop("palette", "Spectral")
    alpha = kwargs.pop("alpha", 0.5)

    if ax is None:
        _, ax = plt.subplots()

    hue = DISTANCE_LABEL

    if not is_long_form(df):
        df = wide_to_long(df)

    if concise:
        df = concise_df(df)
        hue = SITE_LABEL
        kwargs.setdefault("style", hue)
        if palette == "Spectral":
            palette = SECTION_PALETTE
            alpha = 1
        if offset:
            # note that concise_df returns a copy, so can change values here
            df.loc[df[SITE_LABEL] == TERMINAL_LABEL, VOLTAGE_LABEL] -= offset

    sns.lineplot(
        data=df,
        x=TIME_LABEL,
        y=VOLTAGE_LABEL,
        hue=hue,
        legend=legend,
        palette=palette,
        alpha=alpha,
        ax=ax,
        **kwargs,
    )

    if ax_props is not None:
        ax.set(**ax_props)
    if not isinstance(thresh, bool):
        ax.axhline(y=thresh, ls="--", c="k", alpha=0.5, lw=1)
    return ax


def get_pulse_xy(amp, frequency, duration):
    """Return 2 arrays of time and current amplitude, respectively"""
    dt = 0.1
    dur = STIM_PULSE_DUR  # ms of each pulse
    delay = STIM_ONSET  # ms

    delay_idx = int(delay / dt)
    x = np.round(np.arange(0, duration + dt + delay, dt), 2)
    y = np.zeros(shape=x.size)

    if frequency > 0:
        num = duration / 1000 * frequency
        per = np.round(1000 / frequency, 1)  # interval between pulse onsets
        x_mask = (
            np.rint(np.linspace(0, duration, int(num + 1)) / dt).astype(int) + delay_idx
        )
        x_mask_end = x_mask + int(dur / dt)

        for x_start, x_end in zip(x_mask, x_mask_end):
            y[x_start:x_end] = amp
    else:
        y[delay_idx:] = amp

    return x[:-1], y[:-1]


def get_pulse_times(frequency, duration):
    """Return when the pulses occured"""
    dt = 0.1
    delay = STIM_ONSET  # ms

    if frequency <= 0:
        return delay

    delay_idx = int(delay / dt)
    x = np.round(np.arange(0, duration + dt + delay, dt), 2)

    num = duration / 1000 * frequency

    x_mask = (
        np.rint(np.linspace(0, duration, int(num + 1)) / dt).astype(int) + delay_idx
    )

    return x[x_mask][:-1]


def save_fig(name, formats=("png", "pdf"), fig=None):
    if fig is None:
        fig = plt.gcf()
    if not name.startswith("fig_"):
        name = f"fig_{name}"
    logger.info(f"saving {name}")
    name_len = len(name)
    for fmt in formats:
        file_name = get_file_path(name, root="save", ext=fmt)
        logger.info("       " + " " * name_len + f" .{fmt}")
        fig.savefig(file_name, facecolor=fig.get_facecolor(), transparent=True)
    logger.info(f"saved")
