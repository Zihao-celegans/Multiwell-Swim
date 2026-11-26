"""Figure 4: curve traits of the CeDiv strains across levamisole dose, one panel per trait.

Each point is the mean +/- SD (sample SD, ddof = 1) across trials of the per-trial strain mean
(mean trait value of that strain's wells at one dose in one trial), as in Fig 3. Strains missing from a
trial (EG4725 and JU258 are absent from test02) are averaged over the trials that have them; a strain with a
single trial has no error bar.
Traits normalized to each strain's own 0 mM control (the *_delta / *_ratio traits) are not defined at 0 mM
(the control is its own reference), so they are drawn at the treated doses only, as in the H2 analysis.
The *_raw traits (range, t_p<N>, slope_rec on the raw activity) are defined at every dose, 0 mM included.
auc_norm is kept in the exported data but not plotted.

Input: the per-well trait tables written by `Ce_div_set_analysis\\curve_traits.py`
(<trial>\\traits\\curve_traits_ActValS_per_well.csv).

Example (from the trial folders; also writes the per-trial means used):
    python generate_Fig4.py --base_dir "<...>\\My_Passport_6TB_Andersen_Lab_1" --export_data data\\Fig4_data.csv

Redraw from the exported CSV alone:
    python generate_Fig4.py --data data\\Fig4_data.csv
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

TRIALS = ["10102026_CeDiv_Leva_test02", "10212026_CeDiv_Leva_test03", "10312026_CeDiv_Leva_test04"]
CSV_NAME = os.path.join("traits", "curve_traits_ActValS_per_well.csv")
# Same strain order and styling as Fig 3.
STRAINS = ["CX11314", "JU258", "DL238", "MY23", "JT11398", "JU775", "CB4856", "EG4725", "ED3017", "LKC34",
           "MY16", "N2"]
# Same trait lists as Ce_div_set_analysis\broad_sense_H2.R (normalized traits have no 0 mM value).
RAW_TRAITS = ["A0", "auc", "auc_norm", "slope_init", "slope_final",
              "range_raw", "slope_rec_raw", "t_p10_raw", "t_p50_raw", "t_p90_raw"]
NORMALIZED_TRAITS = ["range_delta", "range_ratio", "slope_rec_delta", "slope_rec_ratio",
                     "t_p10_delta", "t_p50_delta", "t_p90_delta",
                     "t_p10_ratio", "t_p50_ratio", "t_p90_ratio",
                     "A0_delta", "auc_delta", "slope_init_delta", "slope_final_delta",
                     "A0_ratio", "auc_ratio", "slope_init_ratio", "slope_final_ratio"]
# Panel order of plot_broad_sense_H2.R (auc_norm is analysed and exported but not plotted)
TRAIT_ORDER = ["A0", "A0_delta", "A0_ratio",
               "auc", "auc_delta", "auc_ratio",
               "slope_init", "slope_init_delta", "slope_init_ratio",
               "slope_final", "slope_final_delta", "slope_final_ratio",
               "range_raw", "range_delta", "range_ratio",
               "t_p10_raw", "t_p10_delta", "t_p10_ratio",
               "t_p50_raw", "t_p50_delta", "t_p50_ratio",
               "t_p90_raw", "t_p90_delta", "t_p90_ratio",
               "slope_rec_raw", "slope_rec_delta", "slope_rec_ratio"]


def load_trial_means(base_dir: str) -> pd.DataFrame:
    """Per trial / strain / dose / trait: mean over wells (NaN values skipped) and the number of wells used."""
    rows = []
    for trial in TRIALS:
        df = pd.read_csv(os.path.join(base_dir, trial, CSV_NAME))
        long = df.melt(id_vars=["strain", "dose_mM"], value_vars=RAW_TRAITS + NORMALIZED_TRAITS,
                       var_name="trait", value_name="value").dropna(subset=["value"])
        long = long[~(long["trait"].isin(NORMALIZED_TRAITS) & (long["dose_mM"] == 0))]
        g = long.groupby(["strain", "dose_mM", "trait"])["value"].agg(value="mean", n_wells="size").reset_index()
        g.insert(0, "trial", trial.rsplit("_", 1)[-1])
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base_dir", help="Folder that contains the trial folders")
    src.add_argument("--data", help="Per-trial means CSV written by --export_data")
    p.add_argument("--export_data", default=None, help="Save the per-trial strain means to a CSV (with --base_dir)")
    p.add_argument("--traits", nargs="+", default=TRAIT_ORDER, help="Traits to plot, in panel order (default: all)")
    p.add_argument("--ncols", type=int, default=3, help="Panels per row")
    p.add_argument("--panel_size", type=float, nargs=2, default=[2.4, 1.7], metavar=("W", "H"),
                   help="Size of one panel incl. labels, inches")
    p.add_argument("--dodge", type=float, default=0.5,
                   help="Total horizontal spread of the strains around each dose, as a fraction of the dose spacing")
    p.add_argument("--legend_ncol", type=int, default=6, help="Strain legend entries per row")
    p.add_argument("--legend_height", type=float, default=0.6, help="Space reserved on top for the legend, inches")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--also_png", action="store_true", help="Also save a PNG next to the SVG")
    p.add_argument("--output_prefix", default="Fig4", help="Writes <prefix>.svg")
    args = p.parse_args()

    if args.data:
        means = pd.read_csv(args.data)
    else:
        means = load_trial_means(args.base_dir)
        if args.export_data:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_data)), exist_ok=True)
            means.to_csv(args.export_data, index=False, float_format="%.6g")

    unknown = set(args.traits) - set(means["trait"])
    if unknown:
        raise SystemExit(f"Unknown trait(s): {sorted(unknown)}")

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
                         "svg.fonttype": "none", "mathtext.default": "regular"})
    cmap = plt.get_cmap("tab10")
    handler_map = {ErrorbarContainer: HandlerErrorbar(yerr_size=0.4)}
    doses = sorted(means["dose_mM"].unique())
    # Strains are spread symmetrically around each dose; --dodge is the total spread as a fraction of the dose spacing.
    spacing = min(b - a for a, b in zip(doses, doses[1:]))
    step = args.dodge * spacing / (len(STRAINS) - 1)
    half_spread = args.dodge * spacing / 2

    n_panels = len(args.traits)
    nrows = math.ceil(n_panels / args.ncols)
    fig, axes = plt.subplots(nrows, args.ncols, figsize=(args.panel_size[0] * args.ncols,
                                                         args.panel_size[1] * nrows + args.legend_height),
                             squeeze=False)
    flat = axes.ravel()

    for ax, trait in zip(flat, args.traits):
        sub = means[means["trait"] == trait]
        for idx, strain in enumerate(STRAINS):
            # trials x doses of per-trial strain means; mean / SD across trials (SD needs >= 2 trials)
            tab = sub[sub["strain"] == strain].pivot(index="trial", columns="dose_mM", values="value")
            if tab.empty:
                continue
            ax.errorbar(tab.columns + (idx - (len(STRAINS) - 1) / 2) * step, tab.mean(axis=0),
                        yerr=tab.std(axis=0, ddof=1), color=cmap(idx % cmap.N),
                        marker="os"[idx // cmap.N], linestyle="none", markersize=3, capsize=2,
                        markerfacecolor="white" if idx >= cmap.N else cmap(idx % cmap.N),  # hollow squares
                        markeredgewidth=0.9, elinewidth=0.75, label=strain, zorder=3)
        ax.set_title(trait, pad=2)
        ax.set_xticks(doses)
        ax.set_xticklabels([f"{d:g}" for d in doses])
        pad = half_spread + 0.05 * (doses[-1] - doses[0])
        ax.set_xlim(doses[0] - pad, doses[-1] + pad)
        ax.tick_params(direction="in", pad=2)
        ax.margins(y=0.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # x label only on the lowest panel of each column
    for i, ax in enumerate(flat[:n_panels]):
        if i + args.ncols >= n_panels:
            ax.set_xlabel("Levamisole (mM)", labelpad=1)
    for ax in flat[n_panels:]:
        ax.axis("off")

    # Strain legend across the top, in rows of --legend_ncol entries
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Strain", loc="upper center", ncol=args.legend_ncol, frameon=False,
               fontsize=8, title_fontsize=8, handlelength=1.5, labelspacing=0.2, columnspacing=1.0,
               borderaxespad=0.1, handler_map=handler_map)
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.8, rect=(0, 0, 1, 1 - args.legend_height / fig.get_figheight()))
    fig.savefig(args.output_prefix + ".svg", dpi=args.dpi)
    print(f"Saved {args.output_prefix}.svg")
    if args.also_png:
        fig.savefig(args.output_prefix + ".png", dpi=args.dpi)
        print(f"Saved {args.output_prefix}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
