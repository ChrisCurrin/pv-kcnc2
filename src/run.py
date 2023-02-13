from itertools import product
from typing import List, Union
import logging
import pandas as pd
from neuron import h
from tqdm import tqdm

from src.constants import (
    CURRENT_LABEL,
    DISTANCE_LABEL,
    KVMUT_FRAC_LABEL,
    SECTION_LABEL,
    STIM_FREQ_LABEL,
    TIME_LABEL,
)
from src.cells.pv_nrn import get_pv, mut, set_nrn_prop, set_stim
from src.data.dataframes import wide_to_long, get_cached_df
from src.data.files import get_file_path
from src.nrn_helpers import remove_cell_from_neuron
from src.utils import get_key

logger = logging.getLogger(__name__)


def run_sims(
    pv,
    stims,
    fractions,
    dur,
    load=False,
    arrow=False,
    shape_plot=True,
    pv_props=None,
    pbar_prefix="",
):
    """
    Run simulations for a given set of parameters

    :param pv: name of the cell to simulate
    :param stims: list of tuples of (current, frequency)
    :param fractions: list of fractions of Kv mutations
    :param dur: duration of the simulation
    :param load: load the results from file if they exist
    :param arrow: save the results in arrow format
    :param shape_plot: plot the shape of the AP
    :param pv_props: properties of the cell mechanisms to set
    :param pbar_prefix: prefix for the progress bar
    :return: dictionary of results
    """

    # create pv if object not passed
    # this is useful if we want to dispose of the object after the function call
    _created_pv = False
    if isinstance(pv, str):
        pv = get_pv(pv)
        _created_pv = True

    if pv_props is None:
        pv_props = {}

    # optionally set the mechanism properties
    # note, mech props are independent of pv.biophys()
    if pv_props:
        try:
            pbar_prefix += "|usetable=0|"
            h.usetable_Kv3 = 0
            h.usetable_Kv3m = 0
        except AttributeError:
            pass
        for key, val in pv_props.items():
            set_nrn_prop(pv, key, val, ignore_error=True)
    else:
        h.usetable_Kv3 = 1
        h.usetable_Kv3m = 1
    if pbar_prefix:
        pbar_prefix += "|>"

    # note that we 'tuple' the product generator to convert it to an iterable of known length for the progressbar
    pbar = tqdm(tuple(product(stims, fractions)), leave=pbar_prefix == "")

    results = {}

    for stim, frac in pbar:
        amp, freq = stim
        key_name = get_key(pv, frac, stim, dur)
        pbar.set_description(f"{pbar_prefix}{key_name}")

        path = get_file_path(key_name)
        long_format_path = get_file_path(key_name, ext="arrow")

        x_df = None
        if not path.exists() or "test" in path.name:
            pbar.set_description(f"{pbar_prefix}{key_name} running")

            pv.biophys()
            mut(pv, frac)

            # run sim and save results
            AP, x_df = get_cached_df(
                key_name, pv, amp, dur, stim_freq=freq, shape_plot=shape_plot
            )

        if arrow and not long_format_path.exists():
            """Save in .feather format, to be loaded using vaex and arrow"""
            if x_df is None:
                # load results
                pbar.set_description(f"{key_name} loading")
                AP, x_df = get_cached_df(key_name)
            # format data
            long_df = wide_to_long(x_df)
            # add metadata as columns with uniform data along the rows
            long_df[KVMUT_FRAC_LABEL] = frac
            long_df[CURRENT_LABEL] = amp
            long_df["Stim. duration"] = dur
            long_df[STIM_FREQ_LABEL] = freq
            long_df["key"] = key_name

            for key, val in AP.items():
                if isinstance(val, list):
                    ap_val = " ".join([str(v.n) for v in val])
                else:
                    ap_val = val.n
                long_df[f"AP_{key}"] = ap_val

            for key, val in pv_props.items():
                long_df[key] = val

            # save
            pbar.set_description(f"{key_name} saving")
            long_df.to_feather(long_format_path)

        if load:
            if x_df is None:
                # load results
                pbar.set_description(f"{key_name} loading")
                AP, x_df = get_cached_df(key_name)
            df = wide_to_long(x_df) if x_df is not None else None

            # store in dict
            results[key_name] = {
                "df": df,
                KVMUT_FRAC_LABEL: frac,
                CURRENT_LABEL: amp,
                "Stim. duration": dur,
                STIM_FREQ_LABEL: freq,
                "APCount": AP,
                **pv_props,
            }

        pbar.set_description(f"{pbar_prefix}{key_name} done")

    if _created_pv:
        remove_cell_from_neuron(pv)

    return results


def record_var(sec, to_record: str, loc: float = 0.5):
    """record voltage ('v'), time ('t') or action potentials ('apc')"""
    to_record = to_record.upper()
    v = h.Vector()
    if to_record == "V":
        v.record(sec(loc)._ref_v)
    elif to_record == "T":
        v.record(h._ref_t)
    elif to_record == "APC":
        v = h.APCount(sec(loc))
    else:
        raise RuntimeError(
            f"unknown recording type '{to_record}' passed to `record_var`"
        )
    return v


def get_trace(
    nrn_cell,
    stim_amp: float,
    stim_dur: float,
    stim_freq: float = 0,
    shape_plot: bool = False,
):
    """Get voltage trace of a neuron.

    If `shape_plot=True`, then the voltage is recorded at the soma and all along the axon, which is captured in
    the pandas `v_df` DataFrame object.
    """

    # must keep reference to object as NEURON automatically clears objects not in scope
    stim = set_stim(nrn_cell.soma[0], stim_amp, stim_dur, frequency=stim_freq)

    t = record_var(nrn_cell.soma[0], "T")
    v = record_var(nrn_cell.soma[0], "V")
    v_df = None  # if shape_plot, then will be a DataFrame

    try:
        APsoma = record_var(nrn_cell.soma[0], "APC", loc=0.5)
        APinit = record_var(nrn_cell.axon[-1], "APC", loc=1)
        APcomm = record_var(nrn_cell.node[-1], "APC", loc=1)
        APprops = [record_var(node_sec, "APC", loc=1) for node_sec in nrn_cell.node]

        AP = {"soma": APsoma, "init": APinit, "comm": APcomm, "props": APprops}

    except AttributeError:
        AP = record_var(nrn_cell.soma[0], "APC", loc=0.5)  # as in original file

    if shape_plot:
        v_rec = []
        x = []
        sec_names = []
        # set distance reference point
        h.distance(0, nrn_cell.soma[0](0.5))

        for seclist in [nrn_cell.somatic, nrn_cell.axonal]:
            for sec in seclist:
                sec_name = sec.hname()
                for seg in sec:
                    v_rec.append(record_var(sec, "V", loc=seg.x))
                    x.append(h.distance(seg))
                    # sec name for every *segment*
                    sec_names.append(sec_name[sec_name.find(".") + 1 :])

    hRun(stim_dur + 20)  # add stim delay

    if shape_plot:
        # columns are distance and index is time
        v_df = pd.DataFrame(
            {
                (sec_name, d_val): v_vec.as_numpy().copy()
                for sec_name, d_val, v_vec in zip(sec_names, x, v_rec)
            },
            index=t.as_numpy().copy(),
            dtype=float,
        )
        v_df.index.name = TIME_LABEL
        v_df.columns.names = [SECTION_LABEL, DISTANCE_LABEL]

    return t, v, AP, v_df


def getIF(
    inputs: List[float], Pv, dur: float = 500, ap_secs: Union[List, str] = "init"
):
    """get input-output (in frequency, Hz) for a list of inputs and diference sections `ap_secs`"""
    if isinstance(ap_secs, str):
        aps = {ap_secs: []}
    else:
        aps = {ap_sec: [] for ap_sec in ap_secs}

    for AMP in inputs:
        ap_dict = get_trace(Pv, AMP, dur)[2]
        # convert to firing rate (Hz)
        for ap_sec, ap_list in aps.items():
            ap_list.append(ap_dict[ap_sec].n * (1000 / dur))
    if len(aps) == 1:
        # only a single section
        return aps[ap_secs]
    return aps


def hRun(T):
    """Run a NEURON simulation for T milliseconds"""
    h.tstop = T
    h.cvode_active(1)
    h.run()
