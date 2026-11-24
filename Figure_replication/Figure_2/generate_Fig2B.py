"""Figure 2B: mean +/- SD activity (ActValS) vs. elapsed time, one line per LEV dose.

Same plot as `activity_by_time.py --by_condition` (the *_line.png figure), redrawn at
journal size (3.5 in wide, Arial 10 pt) with only the x and y axes drawn (no top/right box).

Example (from the raw results.json files; also writes the per-well data CSV):
    python generate_Fig2B.py --data_dir "<...>\\08032026_N2_Levamisole_DRC_test02" --layout "<...>\\dose_layout.csv" --export_data data\\Fig2B_data.csv --output Fig2B.svg

Redraw from the exported CSV alone (no results files or layout needed):
    python generate_Fig2B.py --data data\\Fig2B_data.csv --output Fig2B.svg
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
# activity_by_time.py lives in the repository root, two folders up.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from activity_by_time import (  # noqa: E402
    LINESTYLE_CYCLE, MARKER_CYCLE, discover_results, export_time_series_csv, load_time_points,
)
from plot_activity_by_worm_count import load_layout_from_csv, well_to_condition_map  # noqa: E402


def load_series_from_csv(path: str) -> tuple:
    """Return (conditions in file order, elapsed_min array, {condition: [values per time point]})."""
    conditions, by_time = [], {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            cond = row["condition"]
            if cond not in conditions:
                conditions.append(cond)
            by_time.setdefault(float(row["elapsed_min"]), {}).setdefault(cond, []).append(float(row["value"]))
    # Layout order (descending dose); the CSV rows follow well order instead.
    conditions.sort(key=lambda c: -float(c.split()[0]))
    elapsed = np.array(sorted(by_time))
    return conditions, elapsed, {c: [np.array(by_time[t].get(c, [])) for t in elapsed] for c in conditions}


def load_series_from_results(args) -> tuple:
    layout = load_layout_from_csv(args.layout)
    well_to_group = well_to_condition_map(layout)
    points = load_time_points(discover_results(args.data_dir), args.metric, set(args.censor),
                              well_to_group=well_to_group)
    if args.export_data:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_data)), exist_ok=True)
        export_time_series_csv(points, args.metric, args.export_data, well_to_group=well_to_group)
    conditions = list(layout.keys())
    elapsed = np.array([p["elapsed_min"] for p in points])
    return conditions, elapsed, {
        c: [p.get("values_by_group", {}).get(c, np.array([])) for p in points] for c in conditions}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_dir", help="Folder with the '<video>_output' subfolders (needs --layout)")
    src.add_argument("--data", help="Per-well CSV written by --export_data (replaces --data_dir/--layout)")
    p.add_argument("--layout", help="Layout CSV (condition,well); one line per condition, in file order")
    p.add_argument("--metric", choices=["ActVal", "ActValS"], default="ActValS")
    p.add_argument("--censor", nargs="*", default=[], help="Well labels excluded from every time point")
    p.add_argument("--export_data", default=None, help="Save the per-well values used to a CSV (with --data_dir)")
    p.add_argument("--legend_title", default="LEV dose", help="Legend title (the dose unit is appended)")
    p.add_argument("--size", type=float, nargs="+", default=[3.56, 3.147], metavar="INCHES",
                   help="Figure size in inches: 'W H' (default = Fig2A's size) or 'W' (height = 3/4 W)")
    p.add_argument("--dpi", type=int, default=300, help="Resolution for raster output formats")
    p.add_argument("--also_png", action="store_true", help="Also save a PNG next to --output")
    p.add_argument("--output", default="Fig2B.svg",
                   help="Output path; the format follows the extension (.svg, .png, .pdf)")
    args = p.parse_args()
    if args.data_dir and not args.layout:
        p.error("--data_dir requires --layout")

    # All text: Arial 10 pt; svg.fonttype=none keeps text editable (as in Fig2A).
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                         "legend.title_fontsize": 10, "svg.fonttype": "none", "mathtext.default": "regular"})

    conditions, elapsed, series = (load_series_from_csv(args.data) if args.data
                                   else load_series_from_results(args))
    fig_w, fig_h = (args.size + [None])[:2]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h or 0.75 * fig_w))
    cmap = plt.get_cmap("tab10")
    # Labels carry the unit once, in the legend title (as in Fig2A's "LEV dose (mM)"), to keep the legend narrow.
    unit = conditions[0].rsplit(" ", 1)[-1]
    for idx, cond in enumerate(conditions):
        means = np.array([v.mean() if len(v) else np.nan for v in series[cond]])
        stds = np.array([v.std() if len(v) else np.nan for v in series[cond]])
        ax.errorbar(elapsed, means, yerr=stds, color=cmap(idx % cmap.N),
                    marker=MARKER_CYCLE[(idx // cmap.N) % len(MARKER_CYCLE)],
                    linestyle=LINESTYLE_CYCLE[(idx // cmap.N) % len(LINESTYLE_CYCLE)],
                    linewidth=1, markersize=3, capsize=2, elinewidth=0.75,
                    label=cond.rsplit(" ", 1)[0], zorder=3)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="in")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Activity (A.U.)")
    # Scientific notation (x10^3 offset at the axis top) keeps the y tick labels short.
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    leg = ax.legend(title=f"{args.legend_title} ({unit})", loc="upper right", bbox_to_anchor=(1.0, 1.06),
                    borderaxespad=0, handlelength=1.5,                     labelspacing=0.3, frameon=False,
                                        handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.4)})  # short legend error bars
    leg._legend_box.sep = 6  # points between the title and the first entry
    fig.tight_layout(pad=0.3)

    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")
    if args.also_png:
        png_path = os.path.splitext(args.output)[0] + ".png"
        fig.savefig(png_path, dpi=args.dpi)
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
