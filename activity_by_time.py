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
and prints summary stats plus the overall correlation between elapsed time
and activity.

Usage:
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

from random_layout_generator import ROW_LABELS

TIMESTAMP_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})")


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


def load_time_points(results_paths: list, metric: str, censor: set) -> list:
    """Return a list of dicts, one per video, sorted by recording time:
    {"vidname", "seconds", "elapsed_min", "values": np.ndarray of per-well activity}.
    """
    points = []
    for path in results_paths:
        with open(path) as fh:
            res = json.load(fh)

        vidname = res["Vidname"]
        seconds = parse_recording_seconds(vidname)

        matrix = np.array(res[metric], dtype=np.float64)
        num_row, num_col = matrix.shape
        labels = well_labels(num_row, num_col)
        flat = matrix.flatten()

        keep = np.array([
            (well not in censor) and not np.isnan(val)
            for well, val in zip(labels, flat)
        ])

        points.append({
            "path": path,
            "vidname": vidname,
            "seconds": seconds,
            "values": flat[keep],
            "n_total": len(labels),
            "n_censored": len(censor & set(labels)),
        })

    points.sort(key=lambda p: p["seconds"])
    t0 = points[0]["seconds"]
    for p in points:
        p["elapsed_min"] = (p["seconds"] - t0) / 60.0

    return points


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_activity_by_time(points: list, metric: str, title: str, save_path: str = None) -> None:
    n_groups = len(points)
    fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.3), 6))
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

    labels = [f"{p['elapsed_min']:.0f} min" for p in points]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Elapsed time since first video (min)", fontsize=11)
    ax.set_ylabel(f"{metric} — Active pixels (A.U.)", fontsize=11)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    points = load_time_points(results_paths, args.metric, censor)

    print(f"[ActivityByTime] Found {len(points)} time point(s):")
    for p in points:
        vals = p["values"]
        mean = np.mean(vals) if len(vals) else float("nan")
        std = np.std(vals) if len(vals) else float("nan")
        print(f"  t={p['elapsed_min']:>6.1f} min   {p['vidname']:<28s}  "
              f"used={len(vals):>3}/{p['n_total']} (censored={p['n_censored']})  "
              f"mean={mean:.2f}  std={std:.2f}")

    all_times = np.concatenate([np.full(len(p["values"]), p["elapsed_min"]) for p in points])
    all_values = np.concatenate([p["values"] for p in points])
    pearson_r, pearson_p = stats.pearsonr(all_times, all_values)
    spearman_r, spearman_p = stats.spearmanr(all_times, all_values)
    print(f"\n[ActivityByTime] Correlation: {args.metric} vs. elapsed time")
    print(f"  Pearson  r = {pearson_r:+.3f}  (p = {pearson_p:.4f})")
    print(f"  Spearman rho = {spearman_r:+.3f}  (p = {spearman_p:.4f})")

    save_path = args.output
    if save_path is None and args.save:
        save_path = os.path.join(out_dir, f"activity_by_time_{args.metric}.png")

    title = f"{args.metric} vs. Recording Time  ({len(points)} video(s))"
    plot_activity_by_time(points, args.metric, title, save_path=save_path)


if __name__ == "__main__":
    main()
