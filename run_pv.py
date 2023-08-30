# %% [markdown]
# # PV-IN KCNC2 Kv3.2 A new genetic cause of childhood epilepsy
#

# %% [markdown]
# ## imports and config

# %%

from itertools import product
import logging
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from neuron import h
from tqdm import tqdm

from src.cells.pv_nrn import get_pv, mut
from src.constants import *
from src.constants import (
    CURRENT_LABEL,
    DISTANCE_LABEL,
    FIRING_RATE_LABEL,
    KVMUT_FRAC_LABEL,
    SECTION_LABEL,
    STIM_FREQ_LABEL,
    TIME_LABEL,
)
from src.data import get_cached_df, set_cache_root
from src.nrn_helpers import init_nrn, remove_cell_from_neuron
from src.run import run_sims
from src.settings import *
from src.vis import (
    get_pulse_times,
    get_pulse_xy,
    plot_voltage_trace,
    save_fig,
    set_default_style,
)

logging.basicConfig(level=logging.INFO)

for logger in ["fontTools"]:
    logging.getLogger(logger).setLevel(logging.WARNING)

if platform.system() == "Windows":
    set_cache_root("E:\\.cache\\pv-kcnc2")

init_nrn(celsius=34, v_init=-80)  # as in BBP optimisation

# checks if parameters used during optimisation are the same as those used during
# simulation (method from PV_template.hoc)
h.check_simulator()

set_default_style()

# %% [markdown]
# ## PV Interneuron

pv = get_pv("default", 1000, 30, 1, 60)
mechs = []
for sec in pv.node:
    for seg in sec:
        for mech in seg:
            mechs.append(mech.name())
        break
    break
if "Kv3" in mechs:
    mech_type = "Kv3"
elif "SKv3_1" in mechs:
    mech_type = "SKv3_1"
else:
    raise ValueError("Kv3.2 mechanism not found")

props_to_find_dict = {
    "Kv3": [
        "theta_m",
        "k_m",
        "tau_m0",
        "tau_m1",
        "phi_m0",
        "phi_m1",
        "sigma_m0",
        "sigma_m1",
    ],
    "SKv3_1": ["iv_shift", "iv_gain", "tau_scale", "tau_shift", "tau_gain"],
}
props_to_find = props_to_find_dict[mech_type]
pv_props = {}
pv_mut_props = {}
for sec in pv.node:
    for seg in sec:
        for mech in seg:
            if mech.name() == mech_type:
                for prop in props_to_find:
                    pv_props[prop] = getattr(mech, prop)
            elif mech.name() == mech_type + "m":
                for prop in props_to_find:
                    pv_mut_props[prop] = getattr(mech, prop)
        break
    break

print(f"Mechanism type: {mech_type}")
print(f"Wild-type properties: {pv_props}")
print(f"Mutant properties: {pv_mut_props}")

remove_cell_from_neuron(pv)


# %% [markdown]
# ### F-I curve

# %%
stims = [(amp, 0) for amp in np.round(np.arange(0.0, 4.1, 0.1), 3)]

fractions = [0, 0.10, 0.20, 0.25, 0.3, 0.4, 0.5]

dur = 250

amp_result = run_sims("default", stims, fractions, dur=dur, load=True, arrow=False)

ap_df = pd.DataFrame()

for key, val in amp_result.items():
    nrn_name = key[: key.find("_")]
    frac = val[KVMUT_FRAC_LABEL]
    current = val[CURRENT_LABEL]
    ap_soma = val["APCount"]["soma"].n
    ap_ais = val["APCount"]["init"].n
    ap_axon = val["APCount"]["comm"].n

    ap_df = pd.concat(
        [
            ap_df,
            pd.DataFrame(
                {
                    KVMUT_FRAC_LABEL: frac,
                    CURRENT_LABEL: current,
                    "Neuron": nrn_name,
                    "loc": ["soma", "AIS", "axon"],
                    "Spikes": [ap_soma, ap_ais, ap_axon],
                },
            ),
        ],
        ignore_index=True,
    )
ap_df[FIRING_RATE_LABEL] = ap_df["Spikes"] / (dur / 1000)

with plt.ion():
    sns.lineplot(
        data=ap_df,
        x=CURRENT_LABEL,
        y=FIRING_RATE_LABEL,
        hue="Neuron",
        style="Neuron",
        # markers=["o", "s", "D"],
        palette="Greys_r",
        size="loc",
        size_order=["AIS"],
    )
    plt.show()

# %% [markdown]
# ### Adjust Kv3.2 params

# %%
stims = [(amp, 0) for amp in np.round(np.arange(0.0, 4.1, 0.1), 3)]

# either change params of the original Kv3.2 channel ("down")
# or the mutated Kv3.2 channel ("up")
# the number is the fraction of Kv3.2 that is the mutant channel (0.25 = 25%)
mut_directions = {"up": [0.25], "down": [0]}

num_spaces = 11
params = {}
# params (values are default -> mutant)
for key, val in pv_mut_props.items():
    # if the same, add 20% value either side
    if pv_props[key] == val:
        params[key] = np.round(
            np.linspace(0.8 * pv_props[key], 1.2 * pv_props[key], num_spaces), 2
        )
    else:
        params[key] = np.round(np.linspace(pv_props[key], val, num_spaces), 2)

param_df = pd.DataFrame()
dur = 250
pbar = tqdm(
    list(product(mut_directions.items(), params.items())),
    desc=f"up/down |> params ({len(params)*num_spaces}) |> fraction=PV name_(stim)_duration",
)

for (mut_up_down, fracs), (param_key, param_space) in pbar:
    if isinstance(param_key, tuple):
        actual_params = [f"{pk}_{mech_type}" for pk in param_key]
    else:
        actual_params = [f"{param_key}_{mech_type}"]

    if mut_up_down == "up":
        actual_params = [f"{p}m" for p in actual_params]
    param_name = " ".join(actual_params)

    length = (
        len(param_space) if not np.iterable(param_space[0]) else len(param_space[0])
    )

    for i in range(length):
        if np.iterable(param_space[0]):
            pv_props = {p: v[i] for p, v in zip(actual_params, param_space)}
        else:
            pv_props = {p: param_space[i] for p in actual_params}

        param_val = tuple(pv_props.values())
        if len(param_val) == 1:
            param_val = param_val[0]
        else:
            param_val = " ".join([str(v) for v in param_val])

        # join dict as string
        pv_key = " ".join([f"{k}={v}" for k, v in pv_props.items()])

        param_result = run_sims(
            pv_key,
            stims,
            fracs,
            dur=dur,
            load=True,
            arrow=False,
            shape_plot=False,  # quicker running/saving/loading but no voltage traces
            pv_props=pv_props,  # change params
            pbar_prefix=f"{mut_up_down}|>{param_key}",
        )

        for key, val in param_result.items():
            nrn_name = key[: key.find("_")]
            frac = val[KVMUT_FRAC_LABEL]
            current = val[CURRENT_LABEL]
            ap_soma = val["APCount"]["soma"].n
            ap_ais = val["APCount"]["init"].n
            ap_axon = val["APCount"]["comm"].n

            param_df = pd.concat(
                [
                    param_df,
                    pd.DataFrame(
                        {
                            "param": param_name,
                            "value": param_val,
                            param_key: param_val,
                            "mutation direction": mut_up_down,
                            KVMUT_FRAC_LABEL: frac,
                            CURRENT_LABEL: current,
                            "Neuron": nrn_name,
                            "loc": ["soma", "AIS", "axon"],
                            "Spikes": [ap_soma, ap_ais, ap_axon],
                        },
                    ),
                ],
                ignore_index=True,
            )
param_df[FIRING_RATE_LABEL] = param_df["Spikes"] / (dur / 1000)
param_df


# %%
sns.relplot(
    data=param_df,
    col="param",
    col_wrap=len(params),
    # row="Neuron",
    hue="value",
    palette="Spectral",
    x=CURRENT_LABEL,
    y=FIRING_RATE_LABEL,
    size="Neuron",
    style="loc",
    style_order=["AIS", "axon"],
    kind="line",
    legend="brief",
    # facet_kws=dict(sharex=False),
)

# %%
with sns.plotting_context("poster"):
    fig, axs = plt.subplot_mosaic(
        [
            [
                f"{p}_{mech_type}" for p in props_to_find
            ],  # how normal channels can be impacted
            [
                f"{p}_{mech_type}m" for p in props_to_find
            ],  # how pathological channels can be repaired
        ],
        sharey=True,
        sharex=True,
        figsize=(8, 4),
    )

    base_kwargs = dict(
        x=CURRENT_LABEL,
        y=FIRING_RATE_LABEL,
        # style="loc",
        # style_order=["AIS"],
        size="Neuron",
    )

    for i, (key, ax) in enumerate(axs.items()):
        col = key.replace(f"_{mech_type}m", "").replace(f"_{mech_type}", "")
        pal = (
            sns.blend_palette(["grey", "g"], n_colors=len(params[col]))
            if "Kv3m" in key
            else sns.blend_palette(["r", "k"], n_colors=len(params[col]))
        )
        sns.lineplot(
            data=param_df[
                (param_df[col] == param_df[col].min())
                & (param_df["mutation direction"] == "up")
            ],
            **base_kwargs,
            color="grey",
            ax=ax,
            linestyle="--",
            legend=False,
        )
        sns.lineplot(
            data=param_df[
                (param_df[col] == param_df[col].max())
                & (param_df["mutation direction"] == "down")
            ],
            **base_kwargs,
            color="k",
            ax=ax,
            linestyle="--",
            legend=False,
        )
        sns.lineplot(
            data=param_df[param_df["param"] == key],
            ax=ax,
            hue="value",
            palette=pal,
            **base_kwargs,
            # legend=(i == 0),
            legend=False,
        )
        ax.set_title(key)

plt.show()


# %% [markdown]
# ### Example trace

# %%
amp = 2.0  # nA
dur = 1000  # ms

pv_wt = get_pv("WT")

pv_mut_1 = get_pv("C125Y - 25%")
pv_mut_1.biophys()
mut(pv_mut_1, 0.25)

pv_mut_half = get_pv("C125Y - 50%")
pv_mut_half.biophys()
mut(pv_mut_half, 0.5)

# create datafrome for voltage at nodes
v_df = pd.DataFrame()

for nrn in [
    pv_wt,
    pv_mut_1,
    pv_mut_half,
    # pv_mut_3_4, pv_mut_half, pv_mut_quart
]:
    # set_nrn_prop(nrn, "ek", -85, ignore_error=True)
    AP, x_df = get_cached_df(nrn.name, nrn, amp, dur, shape_plot=True)
    soma = x_df.iloc[:, 0]
    tip = x_df.iloc[:, -1]

    ais_columns = sorted(set([(a, x) for a, x in x_df.columns if "axon" in a]))
    ais_v = x_df[ais_columns[-1]]

    nrn_name = nrn.name[: nrn.name.find("(")].replace("test", "")

    v_df = pd.concat(
        [
            v_df,
            pd.DataFrame(
                {
                    VOLTAGE_LABEL: tip.values,
                    TIME_LABEL: tip.index,
                    "Neuron": nrn_name,
                    "loc": "axon",
                },
            ),
            pd.DataFrame(
                {
                    VOLTAGE_LABEL: soma.values,
                    TIME_LABEL: soma.index,
                    "Neuron": nrn_name,
                    "loc": "soma",
                },
            ),
            pd.DataFrame(
                {
                    VOLTAGE_LABEL: ais_v.values,
                    TIME_LABEL: ais_v.index,
                    "Neuron": nrn_name,
                    "loc": "AIS",
                },
            ),
        ],
        ignore_index=True,
    )

g = sns.relplot(
    data=v_df,
    x=TIME_LABEL,
    y=VOLTAGE_LABEL,
    hue="Neuron",
    hue_order=[
        nrn.name[: nrn.name.find("(")].replace("test", "")
        for nrn in [pv_wt, pv_mut_1, pv_mut_half]
    ],
    palette=["k", "grey", "grey"],
    style="Neuron",
    dashes=["", "", (4.8, 1.8)],
    size="loc",
    size_order=["AIS"],
    legend=False,
    aspect=8,
    height=1,
    row="Neuron",
    kind="line",
)
# remove borders, labels, and ticks
sns.despine(left=True, bottom=True)
g.set(xlabel="", ylabel="", xticks=[], yticks=[])

for nrn in [pv_wt, pv_mut_1, pv_mut_half]:
    remove_cell_from_neuron(nrn)


# %%

fig, ax = plt.subplots(figsize=(8, 1))
sns.lineplot(
    data=v_df,
    x=TIME_LABEL,
    y=VOLTAGE_LABEL,
    hue="Neuron",
    hue_order=[
        nrn.name[: nrn.name.find("(")].replace("test", "") for nrn in [pv_wt, pv_mut_1]
    ],
    palette=["k", "g"],
    style="Neuron",
    # dashes=["", "", (4.8, 1.8)],
    size="loc",
    size_order=["soma", "AIS", "axon"],
    ax=ax,
    legend=False,
)
# remove borders, labels, and ticks
sns.despine(ax=ax, left=True, bottom=True)
# ax.set(xlabel="", ylabel="", xticks=[], yticks=[])
ax.set_xlim(0, 100)


# %%
