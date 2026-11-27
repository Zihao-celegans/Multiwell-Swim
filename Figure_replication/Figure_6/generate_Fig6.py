"""Figure 6: broad-sense heritability (H2) of the curve traits, one panel per trait group, coloured by levamisole dose.

One panel per trait symbol of Fig 5 (3 x 3 panels, letters A-I in the row order of Fig 5). The x axis of a panel is the
version of the trait (raw activity, control-subtracted = delta, control-divided = r); the points at each version are
the H2 at the three doses (colour-coded).
Point = merged H2 of the three trials (test02 + test03 + test04, all 12 strains; lmer(value ~ trial + (1|strain)),
H2 = Vg / (Vg + Ve)), error bar = 5th-95th percentile of the 100-rep within strain x trial bootstrap.
The delta / ratio traits are undefined at 0 mM (the control is its own reference), so they have two doses only.

Input: the merged H2 table written by `Ce_div_set_analysis\\broad_sense_H2_merged.R`
(<merged folder>\\curve_traits_ActValS_merged_broad_sense_H2.csv; rows with strain_set == all_strains).

Example (also writes the plotted values):
    python generate_Fig6.py --base_dir "<...>\\My_Passport_6TB_Andersen_Lab_1" --export_data data\\Fig6_data.csv

Redraw from the exported CSV alone:
    python generate_Fig6.py --data data\\Fig6_data.csv
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

MERGED_DIR = "Merged_CeDiv_Leva_test02_test03_test04"
CSV_NAME = "curve_traits_ActValS_merged_broad_sense_H2.csv"
# One panel per group, in the row order of Fig 5: (mathtext symbol, its own subscript, raw / delta / ratio column).
# The x tick labels are the full symbols, e.g. A_0, A_{0,Δ}, A_{0,r}; the normalization adds Δ (delta) or r (ratio).
GROUPS = [
    (r"\mathit{A}", "0", "A0", "A0_delta", "A0_ratio"),
    (r"\mathit{R}", "", "range_raw", "range_delta", "range_ratio"),
    (r"\mathrm{AUC}", "", "auc", "auc_delta", "auc_ratio"),
    (r"\mathit{k}", r"\mathrm{init}", "slope_init", "slope_init_delta", "slope_init_ratio"),
    (r"\mathit{k}", r"\mathrm{final}", "slope_final", "slope_final_delta", "slope_final_ratio"),
    (r"\mathit{k}", r"\mathrm{rec}", "slope_rec_raw", "slope_rec_delta", "slope_rec_ratio"),
    (r"\mathit{t}", "10", "t_p10_raw", "t_p10_delta", "t_p10_ratio"),
    (r"\mathit{t}", "50", "t_p50_raw", "t_p50_delta", "t_p50_ratio"),
    (r"\mathit{t}", "90", "t_p90_raw", "t_p90_delta", "t_p90_ratio"),
]
# Version of a trait and the subscript it adds to the symbol: raw activity, control-subtracted, control-divided
MODES = [("raw", ""), ("delta", r"\Delta"), ("ratio", r"\mathit{r}")]


def symbol_label(symbol: str, sub: str, mode_sub: str) -> str:
    """Mathtext of a trait symbol, e.g. (k, final, Δ) -> $k_{final,Δ}$."""
    subscript = ",".join(s for s in (sub, mode_sub) if s)
    return f"${symbol}_{{{subscript}}}$" if subscript else f"${symbol}$"
# Dose colours (0, 0.0125, 0.025 mM): colour-blind safe, and not the strain colours of Fig 5
DOSE_COLORS = ["#7F7F7F", "#0072B2", "#CC79A7"]
LETTER_GAP = 2  # pt between the panel letter and the H2 label
Y_LABELPAD = 2  # pt between the tick labels and the H2 label
TAG_SIZE = 12  # pt, panel letters


def load_h2(base_dir: str) -> pd.DataFrame:
    """Merged all-strains H2 of the plotted traits, with the group index and the version (raw / delta / ratio)."""
    df = pd.read_csv(os.path.join(base_dir, MERGED_DIR, CSV_NAME))
    df = df[df["strain_set"] == "all_strains"]
    rows = []
    for g, (_, _, *cols) in enumerate(GROUPS):
        for (mode, _), col in zip(MODES, cols):
            sub = df[df["trait"] == col][["trait", "dose_mM", "H2", "lower_5", "upper_95", "n_strains", "n_wells"]]
            if sub.empty:
                raise SystemExit(f"Trait {col!r} not found in the merged H2 table")
            rows.append(sub.assign(group=g, mode=mode))
    return pd.concat(rows, ignore_index=True)[["group", "mode", "trait", "dose_mM", "H2", "lower_5", "upper_95",
                                              "n_strains", "n_wells"]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base_dir", help="Folder that contains the merged H2 folder")
    src.add_argument("--data", help="H2 CSV written by --export_data")
    p.add_argument("--export_data", default=None, help="Save the plotted H2 values to a CSV (with --base_dir)")
    p.add_argument("--ncols", type=int, default=3, help="Panels per row")
    p.add_argument("--size", type=float, nargs=2, default=[7.5, 7.5], metavar=("W", "H"),
                   help="Size of the whole figure, inches (fixed; nothing is cropped)")
    p.add_argument("--dodge", type=float, default=0.36,
                   help="Total horizontal spread of the three doses around each trait version (x unit = one version)")
    p.add_argument("--col_gap", type=float, default=12.0,
                   help="White space (pt) between the end of a panel's x axis and the panel letter of the next column")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--also_png", action="store_true", help="Also save a PNG next to the SVG")
    p.add_argument("--output_prefix", default="Fig6", help="Writes <prefix>.svg")
    args = p.parse_args()

    if args.data:
        h2 = pd.read_csv(args.data)
    else:
        h2 = load_h2(args.base_dir)
        if args.export_data:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_data)), exist_ok=True)
            h2.to_csv(args.export_data, index=False, float_format="%.6g")

    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                         "svg.fonttype": "none", "mathtext.fontset": "custom", "mathtext.default": "regular",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    doses = sorted(h2["dose_mM"].unique())
    # x = trait version (raw, delta, ratio) at 0, 1, 2; the doses are colour-coded points spread around each version.
    # The points of a version are centred on its tick, so delta / ratio (no 0 mM) get two points, one slot apart;
    # --dodge = the spread of a version with all three doses
    slot = args.dodge / (len(doses) - 1)
    pad = 0.15  # empty x range beyond the outermost points on the right
    pad_left = 0.3  # larger on the left, so the 0 mM points stay clear of the inward y ticks

    nrows = math.ceil(len(GROUPS) / args.ncols)
    fig, axes = plt.subplots(nrows, args.ncols, figsize=args.size, squeeze=False, sharey=True)
    flat = axes.ravel()

    for i, (ax, (symbol, sub_symbol, *_)) in enumerate(zip(flat, GROUPS)):
        for m, (mode, _) in enumerate(MODES):
            sub = h2[(h2["group"] == i) & (h2["mode"] == mode)]
            have = sorted(sub["dose_mM"].unique())
            for k, dose in enumerate(have):
                row = sub[sub["dose_mM"] == dose]
                color = DOSE_COLORS[doses.index(dose)]
                x = m + (k - (len(have) - 1) / 2) * slot
                # Interval from the stored percentiles (drawn as is, even where the point estimate lies just outside it)
                ax.vlines(x, row["lower_5"], row["upper_95"], color=color, linewidth=1.8, zorder=2)
                ax.plot(x, row["H2"], color=color, marker="o", markersize=5.8, linestyle="none", zorder=3)
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels([symbol_label(symbol, sub_symbol, mode_sub) for _, mode_sub in MODES], fontsize=10)
        ax.set_xlim(-args.dodge / 2 - pad_left, len(MODES) - 1 + args.dodge / 2 + pad)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.tick_params(direction="in", pad=2, labelleft=True)  # sharey hides them otherwise; keep them on every panel
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if i == 0:
            # Legend entries as error bars (dummy artists without data), inside panel A where the upper right is empty
            for dose, color in zip(doses, DOSE_COLORS):
                ax.errorbar([np.nan], [np.nan], yerr=0.1, color=color, marker="o", markersize=5.8, linestyle="none",
                            elinewidth=1.8, capsize=0, label=f"{dose:g} mM")
            handles, labels = ax.get_legend_handles_labels()
            # As in Fig 2B: title on top, the doses (with unit) below
            leg = ax.legend(handles, labels, title="LEV dose", loc="upper right", frameon=False, fontsize=10,
                            title_fontsize=10, handlelength=1.0, handletextpad=0.45, labelspacing=0.85,
                            borderaxespad=0, borderpad=0,
                            handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.7)})
            leg._legend_box.sep = 6  # points between the title and the first entry
        ax.set_ylabel(r"$\mathit{H}^{2}$", labelpad=Y_LABELPAD)

    # No bbox_inches="tight": the page size must stay exactly --size.
    fig.tight_layout(pad=0.3, h_pad=0.7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    fig_w_px = fig.bbox.width
    # Panel letters (bold, centred on a common x, left of each H2 label). Measure the widest one and the space the
    # H2 label and tick labels take left of an axes, then re-space the columns: the first letter touches the left page
    # edge, and every column is [letter | H2 label | tick labels | axes] with --col_gap between the end of one axes and
    # the next column's letter, and the same small margin as tight_layout at the right edge.
    letters = []
    for i, ax in enumerate(flat[:len(GROUPS)]):
        letter = ax.annotate(chr(ord("A") + i), xy=(0, 1), xycoords=("figure fraction", "axes fraction"),
                             xytext=(0, 0), textcoords="offset points", ha="center", va="top",
                             fontsize=TAG_SIZE, fontweight="bold")
        letter.set_in_layout(False)
        letters.append(letter)
    letter_w = max(t.get_window_extent(renderer).width for t in letters)
    ax0 = flat[0]
    label_off = ax0.get_position().x0 * fig_w_px - ax0.yaxis.label.get_window_extent(renderer).x0
    left_w = letter_w + LETTER_GAP * px_per_pt + label_off  # letter + gap + (H2 label ... axes) of one column
    gap = args.col_gap * px_per_pt
    right_margin = 3 * px_per_pt  # = tight_layout pad of 0.3 x 10 pt
    axes_w = (fig_w_px - right_margin - (args.ncols - 1) * gap - args.ncols * left_w) / args.ncols
    for i, (ax, letter) in enumerate(zip(flat, letters)):
        start = (i % args.ncols) * (left_w + axes_w + gap)
        pos = ax.get_position()
        ax.set_position(((start + left_w) / fig_w_px, pos.y0, axes_w / fig_w_px, pos.height))
        letter.xy = ((start + letter_w / 2) / fig_w_px, 1)
    # Legend of panel A: raise it so the top of its title is level with the top of the letter A
    fig.canvas.draw()
    # The bbox tops match, but the larger bold letter has more space above its caps: the 0.3 px (figure dpi) offset
    # (checked on the 300 dpi PNG) puts the ink tops of the letter and the legend title level
    shift = letters[0].get_window_extent(renderer).y1 - leg.get_title().get_window_extent(renderer).y1 + 0.0
    ax0_h = flat[0].get_window_extent(renderer).height
    ax0_w = flat[0].get_window_extent(renderer).width
    # +2 px (figure dpi): the legend text ends that far short of the box edge; this puts the "m" of "0.0125 mM" at the
    # right end of the x axis (checked on the 300 dpi PNG)
    leg.set_bbox_to_anchor((1 + 2 / ax0_w, 1 + shift / ax0_h), transform=flat[0].transAxes)
    fig.savefig(args.output_prefix + ".svg", dpi=args.dpi)
    print(f"Saved {args.output_prefix}.svg")
    if args.also_png:
        fig.savefig(args.output_prefix + ".png", dpi=args.dpi)
        print(f"Saved {args.output_prefix}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
