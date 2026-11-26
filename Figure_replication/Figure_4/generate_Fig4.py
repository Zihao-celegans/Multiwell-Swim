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
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.ticker import MaxNLocator, ScalarFormatter

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
# Panel order: each row is the raw / delta / ratio version of one trait (auc_norm is analysed and exported but not plotted)
TRAIT_ORDER = ["A0", "A0_delta", "A0_ratio",
               "range_raw", "range_delta", "range_ratio",
               "auc", "auc_delta", "auc_ratio",
               "slope_init", "slope_init_delta", "slope_init_ratio",
               "slope_final", "slope_final_delta", "slope_final_ratio",
               "slope_rec_raw", "slope_rec_delta", "slope_rec_ratio",
               "t_p10_raw", "t_p10_delta", "t_p10_ratio",
               "t_p50_raw", "t_p50_delta", "t_p50_ratio",
               "t_p90_raw", "t_p90_delta", "t_p90_ratio"]

# Trait symbol (mathtext) and its own subscript; the normalization adds Delta (delta) or r (ratio), raw adds nothing.
SYMBOLS = {"A0": (r"\mathit{A}", "0"), "auc": (r"\mathrm{AUC}", ""),
           "slope_init": (r"\mathit{k}", r"\mathrm{init}"), "slope_final": (r"\mathit{k}", r"\mathrm{final}"),
           "slope_rec": (r"\mathit{k}", r"\mathrm{rec}"), "range": (r"\mathit{R}", ""),
           "t_p10": (r"\mathit{t}", "10"), "t_p50": (r"\mathit{t}", "50"), "t_p90": (r"\mathit{t}", "90")}
MODE_SUBSCRIPTS = {"raw": "", "delta": r"\Delta", "ratio": r"\mathit{r}"}
ROMAN = ["i", "ii", "iii", "iv", "v", "vi"]  # column tags of the panels
TITLE_PAD = 5  # pt between the axes top and the baseline of the trait label / panel tags
NUMERAL_OFFSET = 14  # pt between the column numeral and the y axis
TAG_SIZE = 12  # pt, panel tags (row letters and column numerals)
LETTER_GAP = 4  # pt between the row letters and the column numeral (i)
# Unit of the raw and delta versions, and of the ratio version ("" = unitless)
UNITS = {"A0": ("A.U.", ""), "range": ("A.U.", ""), "auc": ("A.U.\u00b7min", "min"),
         "slope_init": ("A.U.\u00b7min$^{-1}$", "min$^{-1}$"), "slope_final": ("A.U.\u00b7min$^{-1}$", "min$^{-1}$"),
         "slope_rec": ("A.U.\u00b7min$^{-1}$", "min$^{-1}$"), "t_p10": ("min", "min"), "t_p50": ("min", "min"),
         "t_p90": ("min", "min")}


def trait_label(trait: str, exponent: int = None) -> str:
    """Mathtext label with unit of a trait column name, e.g. t_p10_delta -> $t_{10,Δ}$ (min); unknown names are
    returned as is. exponent adds a x10^exponent factor to the unit (values shown divided by 10^exponent).
    """
    base, _, mode = trait.rpartition("_")
    if mode in MODE_SUBSCRIPTS and base in SYMBOLS:
        symbol, sub = SYMBOLS[base]
        subscript = ",".join(s for s in (sub, MODE_SUBSCRIPTS[mode]) if s)
    elif trait in SYMBOLS:
        base, mode = trait, "raw"
        (symbol, subscript) = SYMBOLS[trait]
    else:
        return trait
    unit = UNITS[base][mode == "ratio"]
    if exponent:
        unit = f"\u00d710$^{{{exponent}}}$ {unit}".strip()
    label = f"${symbol}_{{{subscript}}}$" if subscript else f"${symbol}$"
    return f"{label} ({unit})" if unit else label


def panel_numeral(index: int, ncols: int) -> str:
    """Column numeral of the panel, e.g. (iii)."""
    return f"({ROMAN[index % ncols]})"


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
    p.add_argument("--size", type=float, nargs=2, default=[7.5, 10.0], metavar=("W", "H"),
                   help="Size of the whole figure, inches (fixed; nothing is cropped)")
    p.add_argument("--dodge", type=float, default=0.6,
                   help="Total horizontal spread of the strains around each dose, as a fraction of the dose spacing")
    p.add_argument("--start_pad", type=float, default=0.12,
                   help="Empty x range left of the outermost strains of a panel (keeps them off the y axis), as a "
                        "fraction of the dose spacing")
    p.add_argument("--end_pad", type=float, default=0.05,
                   help="Empty x range beyond the outermost strains of a panel, as a fraction of the dose spacing")
    p.add_argument("--col_gap", type=float, default=10.0,
                   help="White space (pt) between the end of a panel's x axis and the numeral of the next column")
    p.add_argument("--legend_ncol", type=int, default=6, help="Strain legend entries per row")
    p.add_argument("--legend_height", type=float, default=0.4, help="Space reserved on top for the legend, inches")
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

    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 10,
                         "svg.fonttype": "none", "mathtext.fontset": "custom", "mathtext.default": "regular",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    cmap = plt.get_cmap("tab10")
    handler_map = {ErrorbarContainer: HandlerErrorbar(yerr_size=0.6)}
    doses = sorted(means["dose_mM"].unique())
    # Strains are spread symmetrically around each dose; --dodge is the total spread as a fraction of the dose spacing.
    spacing = min(b - a for a, b in zip(doses, doses[1:]))
    step = args.dodge * spacing / (len(STRAINS) - 1)
    half_spread = args.dodge * spacing / 2
    end_pad = args.end_pad * spacing
    start_pad = args.start_pad * spacing
    full_span = doses[-1] - doses[0] + 2 * half_spread + start_pad + end_pad  # x range of a panel with every dose

    n_panels = len(args.traits)
    nrows = math.ceil(n_panels / args.ncols)
    fig, axes = plt.subplots(nrows, args.ncols, figsize=args.size, squeeze=False)
    flat = axes.ravel()
    narrow_axes = []
    numeral_by_col = {}
    letters = []

    for i, (ax, trait) in enumerate(zip(flat, args.traits)):
        sub = means[means["trait"] == trait]
        for idx, strain in enumerate(STRAINS):
            # trials x doses of per-trial strain means; mean / SD across trials (SD needs >= 2 trials)
            tab = sub[sub["strain"] == strain].pivot(index="trial", columns="dose_mM", values="value")
            if tab.empty:
                continue
            ax.errorbar(tab.columns + (idx - (len(STRAINS) - 1) / 2) * step, tab.mean(axis=0),
                        yerr=tab.std(axis=0, ddof=1), color=cmap(idx % cmap.N),
                        marker="os"[idx // cmap.N], linestyle="none", markersize=3, capsize=0,
                        markerfacecolor="white" if idx >= cmap.N else cmap(idx % cmap.N),  # hollow squares
                        markeredgewidth=0.9, elinewidth=0.75, label=strain, zorder=3)
        # Trait symbol above the y axis, left-aligned with it
        ax.set_title(trait_label(trait), loc="left", pad=TITLE_PAD, fontsize=10)
        # Ticks only at the doses that have data (no 0 mM tick for the control-normalized traits)
        have = sorted(sub["dose_mM"].unique())
        ax.set_xticks(have)
        ax.set_xticklabels([f"{d:g}" for d in have])
        # Same x scale in every panel: the range is the data extent (outer doses +- the strain spread) plus --start_pad /
        # --end_pad; panels with fewer doses get a shorter axes, resized below to keep the scale.
        lo = have[0] - half_spread - start_pad
        hi = have[-1] + half_spread + end_pad
        ax.set_xlim(lo, hi)
        narrow_axes.append((ax, i % args.ncols, (hi - lo) / full_span))
        ax.tick_params(direction="in", pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))  # 3-4 y ticks in view
        # Scientific notation (factor moved into the label below) for tick values of 1000 or more, or below 0.01
        formatter = ScalarFormatter()
        formatter.set_powerlimits((-3, 3))
        formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.margins(y=0.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # x label only on the lowest panel of each column
    for i, ax in enumerate(flat[:n_panels]):
        if i + args.ncols >= n_panels:
            ax.set_xlabel("LEV dose (mM)", labelpad=1)
        # Column numeral just left of the y axis, level with the trait label (above the top tick label)
        numeral = ax.annotate(panel_numeral(i, args.ncols), xy=(0, 1), xycoords="axes fraction",
                              xytext=(-NUMERAL_OFFSET, TITLE_PAD), textcoords="offset points", ha="right",
                              va="baseline", fontsize=TAG_SIZE)
        numeral.set_in_layout(False)
        numeral_by_col[i % args.ncols] = numeral
        if i % args.ncols == 0:
            # Bold row letter, centred on a common x near the left page edge, level with the panel title
            letter = ax.annotate(chr(ord("A") + i // args.ncols), xy=(0, 1),
                                 xycoords=("figure fraction", "axes fraction"), xytext=(0, TITLE_PAD),
                                 textcoords="offset points", ha="center", va="baseline", fontsize=TAG_SIZE,
                                 fontweight="bold")
            letter.set_in_layout(False)  # it must not widen the axes box; x is set once the layout is known
            letters.append(letter)
    for ax in flat[n_panels:]:
        ax.axis("off")

    # An offset text such as 1e6 would need extra room above the axes: hide it and put the factor in the label
    fig.canvas.draw()
    for i, (ax, trait) in enumerate(zip(flat, args.traits)):
        offset = ax.yaxis.get_offset_text().get_text()
        match = re.fullmatch(r"1e(-?\d+)", offset.replace("\u2212", "-"))
        if match:
            ax.yaxis.get_offset_text().set_visible(False)
            ax.set_title(trait_label(trait, int(match.group(1))), loc="left", pad=TITLE_PAD, fontsize=10)
        elif offset:
            raise SystemExit(f"Unexpected y offset text {offset!r} in panel {trait}")

    # Strain legend across the top, in rows of --legend_ncol entries
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=args.legend_ncol, frameon=False,
               fontsize=10, handlelength=1.5, labelspacing=0.5, columnspacing=1.0,
               borderaxespad=0.1, handler_map=handler_map)
    # No bbox_inches="tight": the page size must stay exactly --size.
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.1, rect=(0, 0, 1, 1 - args.legend_height / fig.get_figheight()))
    # Resize the axes: the x scale is the same in every panel (panels with fewer doses are proportionally shorter), and
    # the white space between the end of an axis and the left edge of the next column's numeral is --col_gap, so the
    # panels use all the width that the numerals and gaps leave; the last axes end at the original right edge.
    # The row letters touch the left page edge (widest letter, others centred on it), LETTER_GAP before the numeral (i).
    col_factor = {c: f for _, c, f in narrow_axes}
    first_row = flat[:args.ncols]
    right = first_row[-1].get_position().x1
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig.bbox.width
    px_per_pt = fig.dpi / 72
    reserved = [(numeral_by_col[c].get_window_extent(renderer).width + NUMERAL_OFFSET * px_per_pt) / fig_w_px
                for c in range(len(first_row))]  # numeral width + its distance to the y axis, as figure fraction
    letter_w = max(letter.get_window_extent(renderer).width for letter in letters)
    for letter in letters:
        letter.xy = (letter_w / 2 / fig_w_px, 1)
    left0 = (letter_w + LETTER_GAP * px_per_pt) / fig_w_px + reserved[0]
    gap = args.col_gap * px_per_pt / fig_w_px  # white space between an axis end and the next numeral, figure fraction
    full_w = (right - left0 - sum(reserved[1:]) - (len(first_row) - 1) * gap) / sum(
        col_factor.get(c, 1.0) for c in range(len(first_row)))  # widest panel; all panels share one x scale
    widths = [full_w * col_factor.get(c, 1.0) for c in range(len(first_row))]
    lefts = [left0]
    for c in range(1, len(first_row)):
        lefts.append(lefts[-1] + widths[c - 1] + gap + reserved[c])
    for ax, c, _ in narrow_axes:
        pos = ax.get_position()
        ax.set_position((lefts[c], pos.y0, widths[c], pos.height))
    fig.savefig(args.output_prefix + ".svg", dpi=args.dpi)
    print(f"Saved {args.output_prefix}.svg")
    if args.also_png:
        fig.savefig(args.output_prefix + ".png", dpi=args.dpi)
        print(f"Saved {args.output_prefix}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
