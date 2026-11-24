"""Figure 3: levamisole activity time courses of the CeDiv strains, one panel (file) per dose.

Writes Fig3A / Fig3B / Fig3C (0, 0.0125, 0.025 mM; each 2.37 in wide, same y range; the strain legend is in Fig3C).
Each point is the mean +/- SD (sample SD, ddof = 1) across trials of the per-trial strain mean
(mean ActValS of that strain's wells at one time point). Strains missing from a trial
(EG4725 and JU258 are absent from test02) are averaged over the trials that have them.
Time points are matched by index (17 per trial, 30 min nominal); the x value is the mean elapsed
time of that index over the trials in the panel.

Input: the per-condition CSVs written by `activity_by_time.py --by_condition --export_csv`
(<trial>\\<dose folder>\\activity_by_time_ActValS_by_condition_data.csv).

Example (from the trial folders; also writes the per-trial means used):
    python generate_Fig3.py --base_dir "<...>\\My_Passport_6TB_Andersen_Lab_1" --export_data data\\Fig3_data.csv

Redraw from the exported CSV alone:
    python generate_Fig3.py --data data\\Fig3_data.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.ticker import MultipleLocator, ScalarFormatter


class ZeroAsIntFormatter(ScalarFormatter):
    """Scientific-offset formatter that prints the zero tick as '0' instead of '0.0'."""

    def __call__(self, x, pos=None):
        return "0" if x == 0 else super().__call__(x, pos)

TRIALS = ["10102026_CeDiv_Leva_test02", "10212026_CeDiv_Leva_test03", "10312026_CeDiv_Leva_test04"]
# (folder name, panel title)
DOSES = [("control", "LEV 0 mM"), ("p0125", "LEV 0.0125 mM"), ("p025", "LEV 0.025 mM")]
# Order and styling follow the earlier per-trial *_line.png plots (tab10 colours, then dashed squares).
STRAINS = ["CX11314", "JU258", "DL238", "MY23", "JT11398", "JU775", "CB4856", "EG4725", "ED3017", "LKC34",
           "MY16", "N2"]
CSV_NAME = "activity_by_time_ActValS_by_condition_data.csv"


def load_trial_means(base_dir: str) -> pd.DataFrame:
    """Per trial / dose / strain / time-point index: mean ActValS over wells and mean elapsed time."""
    rows = []
    for trial in TRIALS:
        for folder, _ in DOSES:
            df = pd.read_csv(os.path.join(base_dir, trial, folder, CSV_NAME))
            # Time-point index = rank of the video time (all strains/wells share a video).
            df["tp"] = df["seconds"].rank(method="dense").astype(int) - 1
            g = df.groupby(["condition", "tp"]).agg(value=("value", "mean"), n_wells=("value", "size"),
                                                    elapsed_min=("elapsed_min", "mean")).reset_index()
            g.insert(0, "dose", folder)
            g.insert(0, "trial", trial.rsplit("_", 1)[-1])
            rows.append(g.rename(columns={"condition": "strain"}))
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base_dir", help="Folder that contains the trial folders")
    src.add_argument("--data", help="Per-trial means CSV written by --export_data")
    p.add_argument("--export_data", default=None, help="Save the per-trial strain means to a CSV (with --base_dir)")
    p.add_argument("--size", type=float, nargs=2, default=[2.37, 3.0], metavar=("W", "H"),
                   help="Size of each panel file, inches (fixed; nothing is cropped)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--also_png", action="store_true", help="Also save PNGs next to the SVGs")
    p.add_argument("--output_prefix", default="Fig3",
                   help="Writes <prefix>A/B/C (one per dose) as .svg")
    args = p.parse_args()

    if args.data:
        means = pd.read_csv(args.data)
    else:
        means = load_trial_means(args.base_dir)
        if args.export_data:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_data)), exist_ok=True)
            means.to_csv(args.export_data, index=False, float_format="%.6g")

    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                         "svg.fonttype": "none", "mathtext.default": "regular"})
    fig_w, fig_h = args.size
    cmap = plt.get_cmap("tab10")
    handler_map = {ErrorbarContainer: HandlerErrorbar(yerr_size=0.4)}

    def save(fig, name):
        # No bbox_inches="tight": the page size must stay exactly fig_w x fig_h.
        fig.savefig(name + ".svg", dpi=args.dpi)
        print(f"Saved {name}.svg")
        if args.also_png:
            fig.savefig(name + ".png", dpi=args.dpi)
            print(f"Saved {name}.png")
        plt.close(fig)

    # Same y range in all panels so the doses can be compared.
    stats = {}
    for folder, _ in DOSES:
        sub = means[means["dose"] == folder]
        x = sub.groupby("tp")["elapsed_min"].mean()
        stats[folder] = (x, {strain: (lambda s: (s.mean(axis=1), s.std(axis=1, ddof=1)))(
            sub[sub["strain"] == strain].pivot(index="tp", columns="trial", values="value").reindex(x.index))
            for strain in STRAINS})
    ymax = max((m + sd.fillna(0)).max() for _, st in stats.values() for m, sd in st.values())
    xmax = max(x.max() for x, _ in stats.values())

    for letter, (folder, title) in zip("ABC", DOSES):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        x, st = stats[folder]
        for idx, strain in enumerate(STRAINS):
            m, sd = st[strain]
            ax.errorbar(x, m, yerr=sd, color=cmap(idx % cmap.N), marker="os"[idx // cmap.N],
                        linestyle=["-", "--"][idx // cmap.N], linewidth=1, markersize=3, capsize=2,
                        elinewidth=0.75, label=strain, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Time (min)", labelpad=1)
        ax.set_ylabel("Activity (A.U.)", labelpad=1)
        ax.set_ylim(-0.03 * ymax, 1.03 * ymax)
        ax.set_xlim(-0.03 * xmax, 1.03 * xmax)
        formatter = ZeroAsIntFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_locator(MultipleLocator(3000))  # 0, 0.3, ... x10^4: one decimal at most
        ax.xaxis.set_major_locator(MultipleLocator(150))
        ax.tick_params(direction="in", pad=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if letter == "C":
            # Upper right is empty once the treated strains have dropped; 8 pt keeps 12 entries short enough.
            ax.legend(title="Strain", loc="upper right", bbox_to_anchor=(1.0, 0.97), frameon=False, fontsize=8,
                      title_fontsize=8, handlelength=1.5, labelspacing=0.2, borderaxespad=0,
                      handler_map=handler_map)
        fig.tight_layout(pad=0.3)
        save(fig, f"{args.output_prefix}{letter}")


if __name__ == "__main__":
    main()
