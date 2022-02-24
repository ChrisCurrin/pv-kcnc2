# helper functions

from numbers import Number
from typing import Iterable
import pandas as pd
import numpy as np

from src.constants import DISTANCE_LABEL, NAV_FRAC_LABEL, SECTION_LABEL


def get_key(pv, frac, nav_loc, stim, dur):
    return f"{pv.name}_{frac}_{nav_loc}_{stim}_{dur}"


def format_nav_loc(nav_loc: str):
    if not isinstance(nav_loc, str) and isinstance(nav_loc, Iterable):
        return "+".join([format_nav_loc(nv) for nv in nav_loc])
    elif nav_loc == "ais":
        return "AIS"
    elif nav_loc == "somatic":
        return format_nav_loc("soma")
    elif nav_loc == "axonal":
        return format_nav_loc(("ais", "nodes"))
    else:
        return nav_loc.capitalize()


def perc_decrease(_df, fp=2):
    if isinstance(_df, pd.DataFrame) and NAV_FRAC_LABEL in _df.columns:
        return np.round(100 * (1 - _df[NAV_FRAC_LABEL]), fp)
    else:
        return np.round(100 * (1 - _df), fp)


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
    return arr[idx], idx


def get_last_sec(long_df):
    return long_df.loc[long_df[DISTANCE_LABEL].idxmax()][SECTION_LABEL]
