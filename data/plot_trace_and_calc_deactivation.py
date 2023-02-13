from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pyabf
import seaborn as sns
from nrnutils import Section, h
from scipy.optimize import curve_fit

from src.measure import calc_deactivation_time_constant_tau


def exp_func_to_fit(x, a, b, c, d):
    return a * np.exp(-1 * (x - b) / c) + d


files = list(Path(".").glob("*4758.abf"))

assert len(files) == 1, "Could not find file"
abf = pyabf.ABF(files[0])
# abf.headerLaunch()
fig, axes = plt.subplots(
    nrows=3, gridspec_kw={"height_ratios": [3, 1, 2]}, figsize=(8, 8)
)
sns.set_palette("Spectral", n_colors=len(abf.sweepList))
dummy_seg = Section(0.1, 0.05, name="dummy_seg")
vclamp_dummy = h.SEClamp(dummy_seg(0.5))
vclamp_dummy.dur1 = 10
vclamp_dummy.dur2 = 36
vclamp_dummy.dur3 = 80
taus = []
infos = []
pal = sns.color_palette("Spectral", n_colors=len(abf.sweepList))

for i in abf.sweepList:
    abf.setSweep(i)
    t_np = abf.sweepX * 1000
    decay_idx = np.where(
        (t_np > (vclamp_dummy.dur2 + vclamp_dummy.dur1))
        & (t_np <= (vclamp_dummy.dur2 + vclamp_dummy.dur1 + vclamp_dummy.dur3))
    )[0]
    popt, pcov = curve_fit(
        exp_func_to_fit,
        t_np[decay_idx],
        abf.sweepY[decay_idx],
        p0=[np.max(abf.sweepY[decay_idx]), 0.1, 2, np.min(abf.sweepY[decay_idx])],
        bounds=([-np.inf, -np.inf, 0, -np.inf], [np.inf, np.inf, 100, np.inf]),
    )
    # taus.append(tau)
    if np.isinf(pcov[0, 0]):
        continue
    infos.append((abf.sweepC[decay_idx][0], *popt))
    tau = popt[2]
    tau = calc_deactivation_time_constant_tau(
        abf.sweepY, abf.sweepX * 1000, vclamp_dummy
    )
    # compare tau calculated from exponential decay index vs curve fit
    print(f"{tau=:.2f} | {popt[2]=:.2f}")
    taus.append((abf.sweepC[decay_idx][0], tau))

    axes[0].plot(
        abf.sweepX * 1000, abf.sweepY, label=f"{tau:.2f}", c=pal[i]  # type: ignore
    )
    axes[1].plot(abf.sweepX * 1000, abf.sweepC, c=pal[i])  # type: ignore
    axes[0].plot(
        t_np[decay_idx],
        exp_func_to_fit(t_np[decay_idx], *popt),
        ls="--",
        lw=1,
        c=pal[i],  # type: ignore
    )

axes[0].legend()
axes[0].set_xlim(45, 150)
axes[1].set_xlim(45, 150)

axes[-1].plot(*zip(*taus), "k-")
for i, (v, tau) in enumerate(taus):
    axes[-1].plot(v, tau, "o", c=pal[i])
# axes[0].set_ylim(0,400)

print(infos)

plt.show()
