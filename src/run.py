from typing import List, Union
import logging
import numpy as np
import pandas as pd
from neuron import h

from src.constants import DISTANCE_LABEL, SECTION_LABEL, TIME_LABEL
from src.settings import STIM_ONSET, STIM_PULSE_DUR

logger = logging.getLogger(__name__)


def mut(Pv, MUT):
    """change Nav1.1 conductance within 'mutated' Nav11m channels"""
    for sec in Pv.all:
        if "Nav11" in [mech.name() for seg in sec.allseg() for mech in seg]:
            gNav = sec(0.5).gNav11bar_Nav11
        for seg in sec:
            for mech in seg:
                if mech.name() == "Nav11m":
                    seg.gNav11bar_Nav11m = MUT * gNav
                    seg.mh_Nav11m, seg.hh_Nav11m = -26.6, -60.2
                    seg.tmh_Nav11m, seg.thh_Nav11m = -40.0, -65.0
                if mech.name() == "Nav11":
                    seg.gNav11bar_Nav11 = (1.0 - MUT) * gNav
    return Pv


def set_relative_nav11bar(
    Pv, proportion: float, at="nodes", base: Union[float, str] = "axon"
):
    """Set Nav1.1 conductance relative to a `base` at section(s) - `at`"""
    if isinstance(base, str):
        assert (
            at != base
        ), f"cannot change values at '{at}' when it is also the `base_sec`."
        base = getattr(Pv, base)[-1].gNav11bar_Nav11

    for sec in getattr(Pv, at):
        if "myelin" in sec.hname():
            continue
        sec.gNav11bar_Nav11 = base * proportion


def set_nrn_prop(pv, property: str, value: float, secs="all", ignore_error=False):
    """set neuron property"""
    for sec in getattr(pv, secs):
        try:
            setattr(sec, property, value)
        except AttributeError as err:
            if not ignore_error:
                raise err


def set_stim(
    sec, amplitude: float, duration: float, frequency: Union[float, bool] = False
):
    """set stimulation

    Either pulse input using Ipulse2 (see mod file for details) or IClamp (see NEURON docs).
    """
    if frequency > 0:
        ipulse = h.Ipulse2(sec(0.5))
        ipulse.dur = STIM_PULSE_DUR  # ms of each pulse
        ipulse.delay = STIM_ONSET  # ms
        ipulse.num = np.ceil(duration / 1000 * frequency)
        ipulse.amp = amplitude  # nA
        ipulse.per = 1000 / frequency  # ms interval between pulse onsets
        return ipulse
    else:
        Ic = h.IClamp(sec(0.5))
        Ic.dur, Ic.delay, Ic.amp = duration, STIM_ONSET, amplitude
        return Ic


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

