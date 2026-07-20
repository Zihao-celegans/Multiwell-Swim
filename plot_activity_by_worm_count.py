"""
plot_activity_by_worm_count.py — Plot activity distributions grouped by worm count (N).

Combines:
  - Well activity values (ActVal / ActValS) from one or more
    activity_analysis.py results JSON files.
  - The well → worm-count assignment produced by random_layout_generator.py
    (either regenerated from the same --seed, or loaded from a saved CSV).

For each worm-count condition N (5, 10, ..., 40 by default), this script
collects the activity value of every well assigned to that condition
(across all provided results files, e.g. replicate videos) and plots the
distribution as a strip + box plot, one group per N — similar in style to
Figure 3 of visualize_activity.py, but grouped by worm count instead of by
metric.

Usage:
    # Layout regenerated from the same seed used with random_layout_generator.py
    python plot_activity_by_worm_count.py \\
        --results "video0004_results.json" \\
        --seed 123

    # Layout loaded from a saved CSV (random_layout_generator.py --output layout.csv)
    python plot_activity_by_worm_count.py \\
        --results "video0004_results.json" "video0005_results.json" \\
        --layout layout.csv \\
        --metric ActVal --save
"""

import argparse
import csv
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt

from random_layout_generator import ROW_LABELS, NUM_COLUMNS, generate_layout


# ---------------------------------------------------------------------------
# Layout loading
# ---------------------------------------------------------------------------

def load_layout_from_csv(csv_path: str) -> dict[int, list[str]]:
    """Load a well→worm-count layout previously saved by
    random_layout_generator.py's write_layout_csv().
    """
    layout: dict[int, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            worm_count = int(row["worm_count"])
            layout.setdefault(worm_count, []).append(row["well"])
    return layout


def well_to_worm_count_map(layout: dict[int, list[str]]) -> dict[str, int]:
    """Invert a worm_count → [wells] layout into well → worm_count."""
    mapping: dict[str, int] = {}
    for worm_count, wells in layout.items():
        for well in wells:
            mapping[well] = worm_count
    return mapping


# ---------------------------------------------------------------------------
# Results loading
# ---------------------------------------------------------------------------

def well_labels_for_matrix(num_row: int, num_col: int) -> list[str]:
    """Row-major well labels matching the ActVal/ActValS matrix layout
    (see utils.load_roi docstring: row 0 = 'A', row 1 = 'B', ... ).
    """
    return [
        f"{ROW_LABELS[r]}{c + 1}"
        for r in range(num_row)
        for c in range(num_col)
    ]


def collect_values_by_worm_count(
    results_paths: list[str],
    metric: str,
    well_to_n: dict[str, int],
) -> dict[int, list[float]]:
    """Gather per-well activity values across all results files, grouped by N."""
    values_by_n: dict[int, list[float]] = {}

    for path in results_paths:
        print(f"[PlotByN] Loading results: {path}")
        with open(path) as fh:
            res = json.load(fh)

        matrix = res[metric]              # list-of-lists, shape (num_row, num_col)
        num_row = len(matrix)
        num_col = len(matrix[0])
        labels = well_labels_for_matrix(num_row, num_col)

        flat = [v for row in matrix for v in row]

        for well, val in zip(labels, flat):
            if well not in well_to_n:
                continue   # well not assigned to any worm-count condition
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            worm_count = well_to_n[well]
            values_by_n.setdefault(worm_count, []).append(float(val))

    return values_by_n


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_activity_by_worm_count(
    values_by_n: dict[int, list[float]],
    metric: str,
    title: str,
    save_path: str = None,
) -> None:
    worm_counts = sorted(values_by_n.keys())
    data = [values_by_n[n] for n in worm_counts]

    fig, ax = plt.subplots(figsize=(max(6, len(worm_counts) * 1.1), 6))
    rng = np.random.default_rng(0)   # reproducible jitter

    positions = np.arange(1, len(worm_counts) + 1)

    ax.boxplot(
        data, positions=positions, widths=0.4, patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        boxprops=dict(facecolor="#444444", alpha=0.6),
        whiskerprops=dict(color="#444444"),
        capprops=dict(color="#444444"),
        flierprops=dict(marker=""),
    )

    for pos, values in zip(positions, data):
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(pos + jitter, values, color="steelblue",
                  s=20, alpha=0.7, zorder=3)
        if values:
            median_val = float(np.median(values))
            ax.text(pos + 0.25, median_val,
                    f"n={len(values)}", va="center", fontsize=8, color="dimgray")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in worm_counts])
    ax.set_xlabel("Worms per well (N)", fontsize=11)
    ax.set_ylabel(f"{metric} — Active pixels (A.U.)", fontsize=11)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PlotByN] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot activity distributions grouped by worm count (N).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True, nargs="+",
                        help="One or more results JSON paths from activity_analysis.py "
                             "(e.g. multiple replicate videos sharing the same layout).")
    parser.add_argument("--layout", default=None,
                        help="Path to a layout CSV saved by random_layout_generator.py "
                             "(--output). If omitted, the layout is regenerated with --seed.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used with random_layout_generator.py, "
                             "used to regenerate the layout when --layout is not given.")
    parser.add_argument("--worm_counts", type=int, nargs="+", default=None,
                        help="Worm-count conditions actually used with "
                             "random_layout_generator.py (e.g. 5 10 15 20 25 30). "
                             "Must match exactly, including omitted conditions, "
                             "or the regenerated layout will not line up. "
                             "Defaults to 5,10,...,40 (all 8 conditions).")
    parser.add_argument("--metric", choices=["ActVal", "ActValS"], default="ActValS",
                        help="Which activity metric to plot.")
    parser.add_argument("--save", action="store_true",
                        help="Save the figure as a PNG alongside the first results file.")
    parser.add_argument("--output", default=None,
                        help="Explicit output PNG path (overrides the --save default path).")
    args = parser.parse_args()

    if args.layout:
        print(f"[PlotByN] Loading layout from CSV: {args.layout}")
        layout = load_layout_from_csv(args.layout)
    else:
        kwargs = {"seed": args.seed}
        if args.worm_counts:
            kwargs["worm_counts"] = tuple(args.worm_counts)
        print(f"[PlotByN] Regenerating layout with seed={args.seed}, "
              f"worm_counts={kwargs.get('worm_counts', 'default (5..40)')}")
        layout = generate_layout(**kwargs)

    well_to_n = well_to_worm_count_map(layout)

    values_by_n = collect_values_by_worm_count(args.results, args.metric, well_to_n)

    if not values_by_n:
        raise SystemExit(
            "[PlotByN] No matching wells found — check that --layout/--seed "
            "matches the plate used for these results."
        )

    print("\n[PlotByN] Summary:")
    for n in sorted(values_by_n):
        vals = values_by_n[n]
        print(f"  N={n:>3}  count={len(vals):>3}  "
              f"mean={np.mean(vals):.2f}  std={np.std(vals):.2f}")

    title = f"{args.metric} vs. Worm Count  ({len(args.results)} video(s))"

    save_path = args.output
    if save_path is None and args.save:
        stem = os.path.splitext(os.path.basename(args.results[0]))[0]
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(args.results[0])),
            f"{stem}_{args.metric}_by_N.png",
        )

    plot_activity_by_worm_count(values_by_n, args.metric, title, save_path=save_path)


if __name__ == "__main__":
    main()
