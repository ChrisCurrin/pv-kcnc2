import numpy as np
import pandas as pd
from neuron import h

from src.constants import DISTANCE_LABEL, SECTION_LABEL, TIME_LABEL, VOLTAGE_LABEL
from src.data import is_long_form, wide_to_long
from src.utils import nearest_idx, nearest_value


def calc_tail_current(ik_np: np.ndarray, t_np: np.ndarray, vclamp: h.SEClamp):
    """
    Calculates tail current for a given vclamp value
    """
    tail_current = np.mean(
        ik_np[
            (t_np > (vclamp.dur2 + vclamp.dur1))
            & (t_np < (vclamp.dur2 + vclamp.dur1 + 10))
        ]
    )
    return tail_current


def calc_activation_time_constant_tau(
    ik_np: np.ndarray, t_np: np.ndarray, vclamp: h.SEClamp
):
    """
    Calculates activation time constant for a given vclamp value
    """
    rise_idx = np.where((t_np > vclamp.dur1) & (t_np < (vclamp.dur2 + vclamp.dur1)))[0]
    activation_ik = ik_np[rise_idx]

    # find idx for 1-1/e of max
    idx = nearest_idx(activation_ik, activation_ik[-1] * (1 - np.exp(-1)))
    return t_np[rise_idx[0] + idx] - t_np[rise_idx[0]]


def calc_deactivation_time_constant_tau(
    ik_np: np.ndarray, t_np: np.ndarray, vclamp: h.SEClamp
):
    """
    Calculates deactivation time constant for a given vclamp value

    Example

    ..plot::

        plt.figure()
        plt.plot(
            t_np[decay_idx][int(0.5 // h.dt) :] - t_np[decay_idx[0]],
            ik_np[decay_idx][int(0.5 // h.dt) :],
            lw=1,
            c="r",
        )
        idx = nearest_idx(tail_ik_np_range, tail_ik_np_range[t_idx] * np.exp(-1))
        plt.plot(
            t_np[decay_idx[0]+ idx] - t_np[decay_idx[0]],
            ik_np[decay_idx[0]+ idx],
            "kx",
        )
        plt.suptitle(f"{t_np[decay_idx[0] + idx] - t_np[decay_idx[0]]}")
        plt.savefig("tmp.jpg", facecolor="w")

    """
    decay_idx = np.where(
        (t_np > (vclamp.dur2 + vclamp.dur1))
        & (t_np <= (vclamp.dur2 + vclamp.dur1 + vclamp.dur3))
    )[0]
    # find idx for 1/e of max
    tail_ik_np = ik_np[decay_idx]
    t_idx = int(
        0.5 // h.dt
    )  # slight delay (in ms, converted to index) to check for decay

    tail_ik_np_range = np.abs(tail_ik_np - tail_ik_np[-1])

    idx = nearest_idx(tail_ik_np_range, tail_ik_np_range[t_idx] * np.exp(-1))

    return t_np[decay_idx[0] + idx] - t_np[decay_idx[0]]


def interpolate(points: pd.DataFrame, step_size: float) -> pd.DataFrame:
    """
    Interpolate a point in a line.
    """
    interp_points = points.reindex(
        index=np.arange(points.index.min(), points.index.max() + 1, step_size)
    )
    interp_points = interp_points.interpolate(method="linear", limit_direction="both")
    assert interp_points is not None
    return interp_points


def find_v_half(points: pd.DataFrame):
    # find v 1/2 point for peak ik
    idx = nearest_idx(points["peak ik"].values, points["peak ik"].max() / 2)
    v_half_peak = points.iloc[idx]["v"]
    return idx, v_half_peak


def get_max_propagation(long_df, thresh=-20.0, time=(0.0,)):
    if VOLTAGE_LABEL not in long_df.columns:
        long_df = wide_to_long(long_df)
    if isinstance(time, float) or isinstance(time, int):
        time_mask = long_df[TIME_LABEL] >= time
    else:
        time_mask = long_df[TIME_LABEL] >= time[0]
        if len(time) > 1:
            time_mask = (time_mask) & (long_df[TIME_LABEL] <= time[1])
    try:
        idx = long_df[(long_df[VOLTAGE_LABEL] >= thresh) & time_mask][
            DISTANCE_LABEL
        ].idxmax()
    except ValueError:
        return None, np.nan
    distance = long_df[DISTANCE_LABEL][idx]
    return idx, distance


def _get_ap_times_wide(x_df, thresh, gap_time, sec):
    soma_df = x_df[sec].iloc[:, 0]
    prev_idx = 0
    above_thresh_mask = soma_df >= thresh
    soma_above_thresh_df = soma_df[above_thresh_mask]
    ap_start_times = []
    while (
        idx := soma_above_thresh_df.loc[prev_idx + gap_time :].first_valid_index()
    ) is not None:
        ap_start_times.append(idx)
        prev_idx = idx
    return np.array(ap_start_times)


def _get_ap_times_long(long_df, thresh, gap_time, sec):
    soma_above_thresh_df = long_df[
        (long_df[SECTION_LABEL] == sec) & (long_df[VOLTAGE_LABEL] >= thresh)
    ]
    prev_idx = 0
    ap_start_times = []
    indices = soma_above_thresh_df.index
    prev_time = 0
    for time in soma_above_thresh_df[TIME_LABEL]:
        if time > prev_time + gap_time:
            ap_start_times.append(time)
            prev_time = time

    return np.array(ap_start_times)


def get_ap_times(long_or_wide_df, thresh=0, gap_time=1.0, sec="soma[0]"):
    if is_long_form(long_or_wide_df):
        return _get_ap_times_long(long_or_wide_df, thresh, gap_time, sec=sec)
    else:
        return _get_ap_times_wide(long_or_wide_df, thresh, gap_time, sec=sec)


def calculate_failures(times, other_times, tol=2.0):
    """Return failure times from `times` to `other_times` (within a tolerance `tol`).

    That is, if there's a time `t` in `times` that does not have a corresponding time `t < t' < t+tol` in `other_times`,
    `t` is added to the list of failure times.
    """
    if len(other_times) == 0:
        return np.array(times)
    failure_times = []
    for t in times:
        other_t = nearest_value(other_times, t)
        #         print(f"{t=} | {other_t=}")
        if other_t < t or other_t - t > tol:
            failure_times.append(t)
    return np.array(failure_times)


def total_area(nrn_sec) -> float:
    """Return the total area of a neuron section in square microns (um2)."""
    from neuron import h

    h.finitialize()
    if isinstance(nrn_sec, h.Section):
        return sum([seg.area() for seg in nrn_sec.allseg()])
    else:
        return sum([total_area(sec) for sec in nrn_sec.all])


if __name__ == "__main__":
    from src.nrn_helpers import init_nrn
    from src.cells.pv_nrn import get_pv
    from src.run import get_trace

    init_nrn()

    amp = 0.1  # nA
    dur = 10  # ms

    t, v, AP, x_df = get_trace(get_pv(), amp, dur, shape_plot=True)

    ap_start_times = get_ap_times(x_df)
    long_df = wide_to_long(x_df)
    ap_start_times_long = get_ap_times(long_df)
    assert all(
        ap_start_times == ap_start_times_long
    ), "ap times not the same for a wide and long dataframe!"
