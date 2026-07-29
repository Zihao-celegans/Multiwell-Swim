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

def load_layout_from_csv(csv_path: str) -> dict[str, list[str]]:
    """Load a well->condition layout previously saved by
    random_layout_generator.py's write_layout_csv().

    Supports both the current 'condition,well' schema and the legacy
    'worm_count,well' schema (old CSVs) for backward compatibility.
    """
    layout: dict[str, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        condition_col = "condition" if "condition" in (reader.fieldnames or []) else "worm_count"
        for row in reader:
            layout.setdefault(row[condition_col], []).append(row["well"])
    return layout


def well_to_condition_map(layout: dict[str, list[str]]) -> dict[str, str]:
    """Invert a condition -> [wells] layout into well -> condition."""
    mapping: dict[str, str] = {}
    for condition, wells in layout.items():
        for well in wells:
            mapping[well] = condition
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


def collect_values_by_condition(
    results_paths: list[str],
    metric: str,
    well_to_condition: dict[str, str],
) -> dict[str, list[float]]:
    """Gather per-well activity values across all results files, grouped by condition."""
    values_by_condition: dict[str, list[float]] = {}

    for path in results_paths:
        print(f"[PlotByCondition] Loading results: {path}")
        with open(path) as fh:
            res = json.load(fh)

        matrix = res[metric]              # list-of-lists, shape (num_row, num_col)
        num_row = len(matrix)
        num_col = len(matrix[0])
        labels = well_labels_for_matrix(num_row, num_col)

        flat = [v for row in matrix for v in row]

        for well, val in zip(labels, flat):
            if well not in well_to_condition:
                continue   # well not assigned to any condition
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            condition = well_to_condition[well]
            values_by_condition.setdefault(condition, []).append(float(val))

    return values_by_condition


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _box_strip_plot(
    values_by_condition: dict[str, list[float]],
    order: list[str],
    ylabel: str,
    title: str,
    save_path: str = None,
) -> None:
    """Shared strip + box plot renderer, one group per condition (in `order`)."""
    data = [values_by_condition[c] for c in order]

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 1.1), 6))
    rng = np.random.default_rng(0)   # reproducible jitter

    positions = np.arange(1, len(order) + 1)

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
    ax.set_xticklabels(order)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PlotByN] Saved: {save_path}")
    plt.show()


def plot_activity_by_worm_count(
    values_by_condition: dict[str, list[float]],
    order: list[str],
    metric: str,
    title: str,
    save_path: str = None,
) -> None:
    """Raw per-well activity, grouped by condition."""
    _box_strip_plot(
        values_by_condition,
        order,
        ylabel=f"{metric} — Active pixels (A.U.)",
        title=title,
        save_path=save_path,
    )


def plot_activity_per_worm(
    values_by_condition: dict[str, list[float]],
    order: list[str],
    metric: str,
    title: str,
    save_path: str = None,
) -> None:
    """Per-worm activity (well activity / N), grouped by condition.

    Highlights whether activity scales linearly with worm count or
    saturates/declines at higher densities (e.g. crowding effects).
    Requires each condition name to be numeric (a worm count).
    """
    try:
        per_worm_by_condition = {
            c: [v / float(c) for v in values] for c, values in values_by_condition.items()
        }
    except ValueError as exc:
        raise SystemExit(
            "[PlotByN] Per-worm normalisation requires numeric condition names "
            "(worm counts), e.g. \"5\", \"10\". Use activity_by_time.py --by_condition "
            "for non-numeric conditions (e.g. drug doses)."
        ) from exc
    _box_strip_plot(
        per_worm_by_condition,
        order,
        ylabel=f"{metric} / N — Active pixels per worm (A.U.)",
        title=title,
        save_path=save_path,
    )


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
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Condition names actually used with random_layout_generator.py "
                             "(e.g. 5 10 15 20 25 30), as numeric worm counts. Must match "
                             "exactly, including omitted conditions, or the regenerated "
                             "layout will not line up. Defaults to \"1\"..\"8\".")
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
        if args.conditions:
            kwargs["conditions"] = tuple(args.conditions)
        print(f"[PlotByN] Regenerating layout with seed={args.seed}, "
              f"conditions={kwargs.get('conditions', 'default (1..8)')}")
        layout = generate_layout(**kwargs)

    well_to_condition = well_to_condition_map(layout)

    values_by_condition = collect_values_by_condition(args.results, args.metric, well_to_condition)

    if not values_by_condition:
        raise SystemExit(
            "[PlotByN] No matching wells found — check that --layout/--seed "
            "matches the plate used for these results."
        )

    order = [c for c in layout.keys() if c in values_by_condition]

    print("\n[PlotByN] Summary:")
    for c in order:
        vals = values_by_condition[c]
        n = float(c)
        per_worm_vals = [v / n for v in vals]
        print(f"  N={c:>3}  count={len(vals):>3}  "
              f"mean={np.mean(vals):.2f}  std={np.std(vals):.2f}  "
              f"|  per-worm mean={np.mean(per_worm_vals):.2f}  "
              f"std={np.std(per_worm_vals):.2f}")

    stem = os.path.splitext(os.path.basename(args.results[0]))[0]
    stem_dir = os.path.dirname(os.path.abspath(args.results[0]))

    # ── Figure: raw activity vs. worm count ──────────────────────────────────
    title = f"{args.metric} vs. Worm Count  ({len(args.results)} video(s))"
    save_path = args.output
    if save_path is None and args.save:
        save_path = os.path.join(stem_dir, f"{stem}_{args.metric}_by_N.png")
    plot_activity_by_worm_count(values_by_condition, order, args.metric, title, save_path=save_path)

    # ── Figure: per-worm activity (activity / N) vs. worm count ─────────────
    per_worm_title = f"{args.metric} per Worm vs. Worm Count  ({len(args.results)} video(s))"
    per_worm_save_path = None
    if args.save:
        per_worm_save_path = os.path.join(stem_dir, f"{stem}_{args.metric}_per_worm_by_N.png")
    plot_activity_per_worm(values_by_condition, order, args.metric, per_worm_title, save_path=per_worm_save_path)


if __name__ == "__main__":
    main()
