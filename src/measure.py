import numpy as np

from src.constants import (DISTANCE_LABEL, SECTION_LABEL, TIME_LABEL,
                           VOLTAGE_LABEL)
from src.data import is_long_form, wide_to_long
from src.utils import nearest_value


def get_max_propagation(long_df, thresh=-20., time=(0.,)):
    if VOLTAGE_LABEL not in long_df.columns:
        long_df = wide_to_long(long_df)
    if isinstance(time, float) or isinstance(time, int):
        time_mask = long_df[TIME_LABEL] >= time
    else:
        time_mask = long_df[TIME_LABEL] >= time[0]
        if len(time) > 1:
            time_mask = (time_mask) & (long_df[TIME_LABEL] <= time[1])
    try:
        idx = long_df[(long_df[VOLTAGE_LABEL] >= thresh) &
                      time_mask][DISTANCE_LABEL].idxmax()
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
    while (idx := soma_above_thresh_df.loc[prev_idx+gap_time:].first_valid_index()) is not None:
        ap_start_times.append(idx)
        prev_idx = idx
    return np.array(ap_start_times)


def _get_ap_times_long(long_df, thresh, gap_time, sec):
    soma_above_thresh_df = long_df[(long_df[SECTION_LABEL] == sec) & (
        long_df[VOLTAGE_LABEL] >= thresh)]
    prev_idx = 0
    ap_start_times = []
    indices = soma_above_thresh_df.index
    prev_time = 0
    for time in soma_above_thresh_df[TIME_LABEL]:
        if time > prev_time+gap_time:
            ap_start_times.append(time)
            prev_time = time

    return np.array(ap_start_times)


def get_ap_times(long_or_wide_df, thresh=0, gap_time=1., sec="soma[0]"):
    if is_long_form(long_or_wide_df):
        return _get_ap_times_long(long_or_wide_df, thresh, gap_time, sec=sec)
    else:
        return _get_ap_times_wide(long_or_wide_df, thresh, gap_time, sec=sec)


def calculate_failures(times, other_times, tol=2.):
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


if __name__ == "__main__":
    try:
        from pv_nrn import get_pv
    except ImportError:
        print("must be run from `pv-scn1a` directory")
    from src.run import get_trace
    amp = 0.1  # nA
    dur = 10  # ms

    t, v, AP, x_df = get_trace(get_pv(), amp, dur, shape_plot=True)

    ap_start_times = get_ap_times(x_df)
    long_df = wide_to_long(x_df)
    ap_start_times_long = get_ap_times(long_df)
    assert all(ap_start_times ==
               ap_start_times_long), "ap times not the same for a wide and long dataframe!"
