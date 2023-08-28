import os
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
from neuron import h
from src.settings import STIM_ONSET, STIM_PULSE_DUR

pvParams = namedtuple("pvParams", "target_myelinated_L node_spacing node_length ais_L")


@lru_cache()
def get_pv(
    name="default",
    target_myelinated_L=1000.0,
    node_spacing=30.0,
    node_length=1.0,
    ais_L=60.0,
):
    """Create a parvalbumin-positive interneuron neuron.

    Note that this function is cached to reduce the number of neurons created by repeated calls.

    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    morph_dir = str(Path(f"{dir_path}/morphologies").absolute()).replace("\\", "/")
    try:
        pv = h.pv(
            morph_dir,
            "C210401C.asc",
            target_myelinated_L,
            node_spacing,
            node_length,
            ais_L,
        )
        pv.name = (
            f"{name}({target_myelinated_L}, {node_spacing}, {node_length}, {ais_L})"
        )
    except AttributeError:
        pv_template = Path("PV_template.hoc").absolute()
        if not pv_template.exists():
            pv_template = Path(f"{dir_path}/PV_template.hoc").absolute()
        h.load_file(str(pv_template).replace("\\", "/"))

        pv = get_pv(name, target_myelinated_L, node_spacing, node_length, ais_L)

    return pv


@lru_cache()
def get_pv_mixed(
    name="default",
    target_myelinated_L=1000.0,
    node_spacing=30.0,
    node_length=1.0,
    ais_L=60.0,
):
    """Create a parvalbumin-positive interneuron neuron.

    Note that this function is cached to reduce the number of neurons created by repeated calls.

    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    morph_dir = str(Path(f"{dir_path}/morphologies").absolute()).replace("\\", "/")
    try:
        pv = h.pv_mixed(
            morph_dir,
            "C210401C.asc",
            target_myelinated_L,
            node_spacing,
            node_length,
            ais_L,
        )
        pv.name = (
            f"{name}({target_myelinated_L}, {node_spacing}, {node_length}, {ais_L})"
        )
    except AttributeError:
        pv_template = Path("PV_template_wt_plus_mutant.hoc").absolute()
        if not pv_template.exists():
            pv_template = Path(f"{dir_path}/PV_template_wt_plus_mutant.hoc").absolute()
        h.load_file(str(pv_template).replace("\\", "/"))

        pv = get_pv_mixed(name, target_myelinated_L, node_spacing, node_length, ais_L)

    return pv


def get_pv_params(pv):
    pv_full_name = pv.name
    p0 = pv_full_name.index("(")
    pv_name = pv_full_name[:p0]
    pv_params = pvParams(
        *[float(param) for param in pv_full_name[p0 + 1 : -1].split(", ")]
    )
    return pv_name, pv_params


def mut(Pv, MUT):
    """change Kv3.1/2 conductance within 'mutated' Kv3m channels"""
    for sec in Pv.all:
        if "Kv3" in [mech.name() for seg in sec.allseg() for mech in seg]:
            gKv3 = sec(0.5).gmax_Kv3
        for seg in sec:
            for mech in seg:
                if mech.name() == "Kv3m":
                    seg.gmax_Kv3m = MUT * gKv3
                elif mech.name() == "Kv3":
                    seg.gmax_Kv3 = (1.0 - MUT) * gKv3
                elif mech.name() == "Kv3mixed":
                    seg.gmax_Kv3mixed = MUT * gKv3
    return Pv


def set_relative_prop(
    Pv, prop: str, proportion: float, at="nodes", base: Union[float, str] = "axon"
):
    """Set prop relative to a `base` at section(s) - `at`"""
    if isinstance(base, str):
        assert (
            at != base
        ), f"cannot change values at '{at}' when it is also the `base_sec`."
        base = getattr(getattr(Pv, base)[-1], prop)

    for sec in getattr(Pv, at):
        if "myelin" in sec.hname():
            continue
        setattr(sec, prop, base * proportion)


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


if __name__ == "__main__":
    from src.nrn_helpers import init_nrn

    init_nrn()
    pv1 = get_pv()
    pv_same = get_pv()
    pv_diff = get_pv("test_pv_dif")
    assert pv1 == pv_same, "not the same object as excepted for the same call to pv()"
    assert (
        pv1 != pv_diff
    ), "expected different argument calls to produce different objects!"
