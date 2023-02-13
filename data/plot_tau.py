import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.data.files import get_file_path

tau_df = pd.read_csv(get_file_path("new_tau", "data", ext="csv")).infer_objects()
tau_types = tau_df["tau type"].unique()

fig, axes = plt.subplot_mosaic([list(tau_types)], figsize=(8, 4))  # type: ignore
axes: dict[str, plt.Axes]

for i, tau_type in enumerate(tau_types):
    tau_type_df = tau_df[tau_df["tau type"] == tau_type]
    ax = axes[tau_type]

    # add error bars from "SE" column
    for group in tau_type_df["group"].unique():
        tau_wt_df = tau_type_df[tau_type_df["group"] == group]
        ax.errorbar(
            tau_wt_df["Potential (mV)"], tau_wt_df["tau (ms)"], yerr=tau_wt_df["SE"]
        )

    # draw lines themselves
    sns.lineplot(
        data=tau_type_df,
        x="Potential (mV)",
        y="tau (ms)",
        hue="group",
        ax=ax,
        legend=bool(i == 0),  # type: ignore
    )
    ax.set_title(tau_type)
    ax.axvline(0, ls="--", color="k", alpha=0.5, zorder=-1)
# ax.spines['left'].set_position(('data', 0))

plt.show()
