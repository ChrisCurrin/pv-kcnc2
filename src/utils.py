# helper functions

from numbers import Number
import numpy as np

from src.constants import DISTANCE_LABEL, SECTION_LABEL


def mut_name(frac):
    if frac == 1:
        return "C125Y"
    elif frac == 0:
        return "WT"
    return f"C125Y - {100*frac:.0f}%"


def get_key(pv, frac, stim, dur):
    basename = mut_name(frac)
    pv_name = pv.name
    if "mixed" in pv_name:
        if "C125Y" in basename:
            basename = f"WT+{basename}"

        pv_name = (
            pv_name.replace("-mixed", "").replace("mixed-", "").replace("mixed", "")
        )
    return f"{basename}_{pv_name}_{stim}_{dur}"


def str_to_tuple(s):
    """helper function to convert str to tuple without ast module"""
    return tuple(
        [
            subs.replace("'", "").replace("(", "").replace(")", "")
            for subs in s.split(", ")
        ]
    )


def nearest_idx(arr: np.ndarray, value: Number):
    idx = (np.abs(arr - value)).argmin()
    return idx


def nearest_value(arr: np.ndarray, value: Number):
    return arr[nearest_idx(arr, value)]


def nearest_idx_val(arr: np.ndarray, value: Number):
    idx = nearest_idx(arr, value)
    return idx, arr[idx]


def get_last_sec(long_df):
    return long_df.loc[long_df[DISTANCE_LABEL].idxmax()][SECTION_LABEL]
