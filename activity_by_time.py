"""
activity_by_time.py — Plot activity as a function of video recording time.

Designed for a series of videos of the *same* plate recorded at intervals
(e.g. every ~15 min) to see whether/how measured activity drifts over the
course of a session (e.g. worms settling, evaporation, temperature changes).

For each video's results.json:
  - The recording time-of-day is parsed from the "Vidname" field
    (e.g. "video0005 10-30-43.avi" -> 10:30:43), which is more reliable
    than the "date" field (that is the *processing* timestamp, not the
    recording timestamp).
  - Elapsed time (minutes) is computed relative to the earliest video.
  - Every non-censored, non-NaN well's activity value is collected.

Plots a box + strip plot (one group per video/time point, x-axis = elapsed
minutes) showing the distribution of per-well activity at each time point,
plus a mean ± SD line plot (x-axis to scale in elapsed minutes) showing the
overall temporal trend, and prints summary stats plus the overall
correlation between elapsed time and activity.

Usage:

    # python activity_by_time.py --input_dir "D:\\MultiWell_swim\\Preliminary\\08072026_N2_Pyrantel_DRC_test02" \\
    # --by_condition --layout "C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\Multiwell_swim\\Well_assignment_0804\\Pyrantel_dose_layout.csv" \\
    # --save --censor C5
    
    # Auto-discover all "<video>_output" subfolders under a parent directory
    python activity_by_time.py \\
        --input_dir "D:\\MultiWell_swim\\Preliminary\\07172026_N2_sorter_edge_effect" \\
        --metric ActValS \\
        --censor G10 G11 G12 H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 \\
        --save

    # Or pass explicit results files
    python activity_by_time.py \\
        --results "video0000..._results.json" "video0005..._results.json" \\
        --metric ActValS --save
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from random_layout_generator import ROW_LABELS, generate_layout
from plot_activity_by_worm_count import load_layout_from_csv, well_to_condition_map

TIMESTAMP_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})")
FIGSIZE_4_3 = (8, 6)  # Fixed 4:3 aspect ratio for saved figures


# ---------------------------------------------------------------------------
# Discovery / data loading
# ---------------------------------------------------------------------------

def discover_results(input_dir: str) -> list:
    """Find all '<video_stem>_results.json' files inside '*_output' folders
    directly under input_dir.
    """
    pattern = os.path.join(input_dir, "*_output", "*_results.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"[ActivityByTime] No '*_output/*_results.json' files found under {input_dir}")
    return paths


def parse_recording_seconds(vidname: str) -> int:
    """Parse the HH-MM-SS timestamp embedded in a video filename
    (e.g. 'video0005 10-30-43.avi') into seconds-since-midnight.
    """
    match = TIMESTAMP_RE.search(vidname)
    if not match:
        raise ValueError(f"Could not parse a HH-MM-SS timestamp from Vidname: {vidname!r}")
    hh, mm, ss = (int(g) for g in match.groups())
    return hh * 3600 + mm * 60 + ss


def well_labels(num_row: int, num_col: int) -> list:
    return [f"{ROW_LABELS[r]}{c + 1}" for r in range(num_row) for c in range(num_col)]


def load_time_points(
    results_paths: list,
    metric: str,
    censor: set,
    well_to_group: dict | None = None,
) -> list:
    """Return a list of dicts, one per video, sorted by recording time:
    {"vidname", "seconds", "elapsed_min", "values": np.ndarray of per-well activity}.

    If well_to_group is given (well label -> group key, e.g. from
    random_layout_generator.py's layout), each point also gets a
    "values_by_group" dict mapping group key -> np.ndarray of that
    group's per-well activity values.
    """
    points = []
    prev_seconds = None
    day_offset = 0
    for path in results_paths:
        with open(path) as fh:
            res = json.load(fh)

        vidname = res["Vidname"]
        seconds = parse_recording_seconds(vidname)
        # Vidname only has HH-MM-SS (no date); detect midnight rollover by a
        # drop in seconds-of-day relative to the previous video (assumes
        # results_paths is already in chronological/recording order).
        if prev_seconds is not None and seconds < prev_seconds:
            day_offset += 24 * 3600
        prev_seconds = seconds
        seconds += day_offset

        matrix = np.array(res[metric], dtype=np.float64)
        num_row, num_col = matrix.shape
        labels = well_labels(num_row, num_col)
        flat = matrix.flatten()

        keep = np.array([
            (well not in censor) and not np.isnan(val)
            for well, val in zip(labels, flat)
        ])

        point = {
            "path": path,
            "vidname": vidname,
            "seconds": seconds,
            "values": flat[keep],
            "n_total": len(labels),
            "n_censored": len(censor & set(labels)),
        }

        if well_to_group is not None:
            values_by_group: dict[object, list] = {}
            for well, val, keep_flag in zip(labels, flat, keep):
                if not keep_flag:
                    continue
                group = well_to_group.get(well)
                if group is None:
                    continue
                values_by_group.setdefault(group, []).append(val)
            point["values_by_group"] = {
                group: np.array(vals) for group, vals in values_by_group.items()
            }

        points.append(point)

    points.sort(key=lambda p: p["seconds"])
    t0 = points[0]["seconds"]
    for p in points:
        p["elapsed_min"] = (p["seconds"] - t0) / 60.0

    return points


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_activity_by_time(points: list, metric: str, save_path: str = None) -> None:
    n_groups = len(points)
    fig, ax = plt.subplots(figsize=FIGSIZE_4_3)
    rng = np.random.default_rng(0)

    positions = np.arange(1, n_groups + 1)
    data = [p["values"] for p in points]

    ax.boxplot(
        data, positions=positions, widths=0.4, patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        boxprops=dict(facecolor="#444444", alpha=0.6),
        whiskerprops=dict(color="#444444"),
        capprops=dict(color="#444444"),
        flierprops=dict(marker=""),
    )

    for pos, p in zip(positions, points):
        values = p["values"]
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(pos + jitter, values, color="steelblue", s=20, alpha=0.7, zorder=3)
        if len(values):
            ax.text(pos + 0.25, float(np.median(values)),
                    f"n={len(values)}", va="center", fontsize=8, color="dimgray")

    labels = [f"{p['elapsed_min']:.0f}" for p in points]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Elapsed time since first video (min)", fontsize=11)
    ax.set_ylabel(f"{metric} - Active pixels (A.U.)", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[ActivityByTime] Saved: {save_path}")
    plt.show()


def plot_activity_by_time_grouped(
    points: list,
    metric: str,
    conditions: list,
    save_path: str = None,
) -> None:
    """Box + strip plot of per-well activity vs. elapsed time, with one
    box (plus jittered individual points) per condition at each time point.

    Args:
        points: Time points from load_time_points(..., well_to_group=...),
            each with a "values_by_group" dict.
        conditions: Condition names, in the order they should appear in the
            legend (e.g. the layout's key order — ascending dose, etc.).
    """
    n_groups = len(points)
    n_conditions = len(conditions)
    fig, ax = plt.subplots(figsize=FIGSIZE_4_3)
    cmap = plt.get_cmap("tab10")
    rng = np.random.default_rng(0)

    box_width = 0.7 / n_conditions
    cluster_positions = np.arange(1, n_groups + 1)

    for idx, condition in enumerate(conditions):
        color = cmap(idx % cmap.N)
        offset = (idx - (n_conditions - 1) / 2) * box_width
        positions = cluster_positions + offset

        data = [
            p.get("values_by_group", {}).get(condition, np.array([]))
            for p in points
        ]

        ax.boxplot(
            data, positions=positions, widths=box_width * 0.85, patch_artist=True,
            medianprops=dict(color="white", linewidth=1.5),
            boxprops=dict(facecolor=color, alpha=0.6, edgecolor=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker=""),
        )

        for pos, values in zip(positions, data):
            if not len(values):
                continue
            jitter = rng.uniform(-box_width * 0.3, box_width * 0.3, size=len(values))
            ax.scatter(pos + jitter, values, color=color, s=12, alpha=0.6,
                       zorder=3, edgecolors="none")

        # Proxy artist so the legend shows a swatch per condition
        # (boxplot patches aren't directly usable as legend handles).
        ax.plot([], [], color=color, marker="s", linestyle="", markersize=8, label=condition)

    labels = [f"{p['elapsed_min']:.0f}" for p in points]
    ax.set_xticks(cluster_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Elapsed time (min)", fontsize=11)
    ax.set_ylabel(f"{metric} - Active pixels (A.U.)", fontsize=11)
    ax.legend(title="Condition", loc="best")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"[ActivityByTime] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Mean ± SD line plots
# ---------------------------------------------------------------------------

def plot_activity_by_time_line(points: list, metric: str, save_path: str = None) -> None:
    """Mean ± SD line plot of per-well activity vs. elapsed time (pooled
    across all wells). X-axis is to scale in elapsed minutes, unlike the
    box plot's evenly-spaced categorical positions, so it accurately shows
    time gaps between videos.
    """
    elapsed = np.array([p["elapsed_min"] for p in points])
    means = np.array([np.mean(p["values"]) if len(p["values"]) else np.nan for p in points])
    stds = np.array([np.std(p["values"]) if len(p["values"]) else np.nan for p in points])

    fig, ax = plt.subplots(figsize=FIGSIZE_4_3)

    ax.errorbar(elapsed, means, yerr=stds, color="steelblue", marker="o", linewidth=2,
                capsize=4, elinewidth=1.5, zorder=3)

    ax.set_xlabel("Elapsed time since first video (min)", fontsize=11)
    ax.set_ylabel(f"{metric} - Active pixels (A.U.)", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[ActivityByTime] Saved: {save_path}")
    plt.show()


def plot_activity_by_time_line_grouped(
    points: list,
    metric: str,
    conditions: list,
    save_path: str = None,
) -> None:
    """Mean ± SD line plot of per-well activity vs. elapsed time, with one
    line (+ shaded SD band) per condition. X-axis is to scale in elapsed
    minutes.

    Args:
        points: Time points from load_time_points(..., well_to_group=...),
            each with a "values_by_group" dict.
        conditions: Condition names, in the order they should appear in the
            legend (e.g. the layout's key order — ascending dose, etc.).
    """
    elapsed = np.array([p["elapsed_min"] for p in points])
    fig, ax = plt.subplots(figsize=FIGSIZE_4_3)
    cmap = plt.get_cmap("tab10")

    for idx, condition in enumerate(conditions):
        color = cmap(idx % cmap.N)
        group_vals = [p.get("values_by_group", {}).get(condition, np.array([])) for p in points]
        means = np.array([np.mean(v) if len(v) else np.nan for v in group_vals])
        stds = np.array([np.std(v) if len(v) else np.nan for v in group_vals])

        ax.errorbar(elapsed, means, yerr=stds, color=color, marker="o", linewidth=2,
                    label=condition, capsize=4, elinewidth=1.5, zorder=3)

    ax.set_xlabel("Elapsed time (min)", fontsize=11)
    ax.set_ylabel(f"{metric} - Active pixels (A.U.)", fontsize=11)
    ax.legend(title="Condition", loc="best")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"[ActivityByTime] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot activity as a function of video recording time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", default=None,
                        help="Parent directory containing multiple '<video>_output' "
                             "subfolders (auto-discovers all '*_results.json' inside them).")
    group.add_argument("--results", nargs="+", default=None,
                        help="Explicit list of results.json paths (one per video/time point).")
    parser.add_argument("--metric", choices=["ActVal", "ActValS"], default="ActValS",
                        help="Which activity metric to plot.")
    parser.add_argument("--censor", nargs="*", default=[],
                        help="Well labels to exclude from every time point "
                             "(e.g. no/insufficient worms), "
                             "e.g. --censor G10 G11 G12 H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12")
    parser.add_argument("--by_condition", action="store_true",
                        help="Group wells by condition (layout produced by "
                             "random_layout_generator.py — worm counts, drug doses, or "
                             "any other named groups) instead of pooling all wells together.")
    parser.add_argument("--layout", default=None,
                        help="Path to a layout CSV saved by random_layout_generator.py "
                             "(--output). If omitted, the layout is regenerated with --seed. "
                             "Condition names/labels are read directly from the CSV. "
                             "Only used with --by_condition.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used with random_layout_generator.py, used to regenerate "
                             "the layout when --layout is not given. Only used with --by_condition.")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Condition names, in order (e.g. --conditions \"0 mM\" \"0.2 mM\" "
                             "\"2 mM\" \"20 mM\"), used to regenerate a layout with "
                             "random_layout_generator.py. Order is preserved in the legend. "
                             "Defaults to the generator's default conditions. Ignored when "
                             "--layout is given (names come from the CSV directly). "
                             "Only used with --by_condition.")
    parser.add_argument("--save", action="store_true",
                        help="Save the figure as a PNG.")
    parser.add_argument("--output", default=None,
                        help="Explicit output PNG path (overrides the --save default path).")
    args = parser.parse_args()

    if args.input_dir:
        results_paths = discover_results(args.input_dir)
        out_dir = args.input_dir
    else:
        results_paths = args.results
        out_dir = os.path.dirname(os.path.abspath(results_paths[0]))

    censor = set(args.censor)

    condition_order = None
    well_to_group = None
    if args.by_condition:
        if args.layout:
            print(f"[ActivityByTime] Loading layout from CSV: {args.layout}")
            layout = load_layout_from_csv(args.layout)
        else:
            kwargs = {"seed": args.seed}
            if args.conditions:
                kwargs["conditions"] = tuple(args.conditions)
            print(f"[ActivityByTime] Regenerating layout with seed={args.seed}, "
                  f"conditions={kwargs.get('conditions', 'default')}")
            layout = generate_layout(**kwargs)

        well_to_group = well_to_condition_map(layout)
        condition_order = list(layout.keys())

    points = load_time_points(results_paths, args.metric, censor, well_to_group=well_to_group)

    print(f"[ActivityByTime] Found {len(points)} time point(s):")
    for p in points:
        vals = p["values"]
        mean = np.mean(vals) if len(vals) else float("nan")
        std = np.std(vals) if len(vals) else float("nan")
        print(f"  t={p['elapsed_min']:>6.1f} min   {p['vidname']:<28s}  "
              f"used={len(vals):>3}/{p['n_total']} (censored={p['n_censored']})  "
              f"mean={mean:.2f}  std={std:.2f}")
        if args.by_condition:
            for condition in condition_order:
                gvals = p["values_by_group"].get(condition, np.array([]))
                gmean = np.mean(gvals) if len(gvals) else float("nan")
                gstd = np.std(gvals) if len(gvals) else float("nan")
                print(f"      {condition:<10s} n={len(gvals):>3}  mean={gmean:.2f}  std={gstd:.2f}")

    all_times = np.concatenate([np.full(len(p["values"]), p["elapsed_min"]) for p in points])
    all_values = np.concatenate([p["values"] for p in points])
    pearson_r, pearson_p = stats.pearsonr(all_times, all_values)
    spearman_r, spearman_p = stats.spearmanr(all_times, all_values)
    print(f"\n[ActivityByTime] Correlation: {args.metric} vs. elapsed time")
    print(f"  Pearson  r = {pearson_r:+.3f}  (p = {pearson_p:.4f})")
    print(f"  Spearman rho = {spearman_r:+.3f}  (p = {spearman_p:.4f})")

    save_path = args.output
    suffix = "_by_condition" if args.by_condition else ""
    line_save_path = None
    if save_path is not None:
        root, ext = os.path.splitext(save_path)
        line_save_path = f"{root}_line{ext}"
    if save_path is None and args.save:
        save_path = os.path.join(out_dir, f"activity_by_time_{args.metric}{suffix}.png")
        line_save_path = os.path.join(out_dir, f"activity_by_time_{args.metric}{suffix}_line.png")

    if args.by_condition:
        plot_activity_by_time_grouped(points, args.metric, condition_order, save_path=save_path)
        plot_activity_by_time_line_grouped(points, args.metric, condition_order, save_path=line_save_path)
    else:
        plot_activity_by_time(points, args.metric, save_path=save_path)
        plot_activity_by_time_line(points, args.metric, save_path=line_save_path)


if __name__ == "__main__":
    main()
