"""
position_effect_analysis.py — Correlate physical distance from the plate
centroid with measured activity, with support for censoring specific wells
(e.g. wells with no/insufficient worms).

For each (non-censored) well:
  - center_distance_px = Euclidean distance (pixels) from that well's ROI
    center to the centroid (mean center) of all ROI wells.
  - activity           = ActVal / ActValS value for that well.

Reports Pearson and Spearman correlation between the two, and plots a
scatter with a linear fit.

Usage:
    python position_effect_analysis.py \\
        --results "video0000 10-14-00_results.json" \\
        --roi     "video0000 10-14-00_roi_info.json" \\
        --metric ActValS \\
        --censor G10 G11 G12 H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 \\
        --save
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from random_layout_generator import ROW_LABELS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activity_matrix(results_path: str, metric: str) -> np.ndarray:
    with open(results_path) as fh:
        res = json.load(fh)
    return np.array(res[metric], dtype=np.float64)   # (num_row, num_col)


def load_roi_centers(roi_path: str) -> tuple:
    with open(roi_path) as fh:
        data = json.load(fh)
    centers = np.array([r["center"] for r in data["roi"]], dtype=np.float64)  # (num_roi, 2)
    return centers, data["num_row"], data["num_col"]


def well_labels(num_row: int, num_col: int) -> list:
    return [f"{ROW_LABELS[r]}{c + 1}" for r in range(num_row) for c in range(num_col)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Correlate distance from plate centroid with activity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True,
                        help="Path to the _results.json file.")
    parser.add_argument("--roi", required=True,
                        help="Path to roi_info.json (needed for physical well centers).")
    parser.add_argument("--metric", choices=["ActVal", "ActValS"], default="ActValS",
                        help="Which activity metric to analyse.")
    parser.add_argument("--censor", nargs="*", default=[],
                        help="Well labels to exclude (e.g. no/insufficient worms), "
                             "e.g. --censor G10 G11 G12 H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12")
    parser.add_argument("--save", action="store_true",
                        help="Save the scatter plot alongside the results file.")
    args = parser.parse_args()

    activity = load_activity_matrix(args.results, args.metric)
    num_row, num_col = activity.shape
    roi_centers, roi_num_row, roi_num_col = load_roi_centers(args.roi)

    if (roi_num_row, roi_num_col) != (num_row, num_col):
        raise SystemExit(
            f"[Position] ROI grid ({roi_num_row}x{roi_num_col}) does not match "
            f"results grid ({num_row}x{num_col})."
        )

    labels = well_labels(num_row, num_col)
    flat_activity = activity.flatten()

    centroid = roi_centers.mean(axis=0)
    center_distance_px = np.linalg.norm(roi_centers - centroid, axis=1)

    censor_set = set(args.censor)
    unknown = censor_set - set(labels)
    if unknown:
        print(f"[Position] WARNING: censored well(s) not found on plate: {sorted(unknown)}")

    keep = np.array([
        (well not in censor_set) and not np.isnan(val)
        for well, val in zip(labels, flat_activity)
    ])

    n_censored = len(censor_set & set(labels))
    n_nan = int(np.isnan(flat_activity).sum())
    print(f"[Position] Total wells: {len(labels)}   "
          f"Censored: {n_censored}   NaN (excluding censored): {n_nan}   "
          f"Used: {int(keep.sum())}")

    x = center_distance_px[keep]
    y = flat_activity[keep]

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    print(f"\n[Position] Correlation: {args.metric} vs. distance from plate centroid")
    print(f"  Pearson  r = {pearson_r:+.3f}  (p = {pearson_p:.4f})")
    print(f"  Spearman rho = {spearman_r:+.3f}  (p = {spearman_p:.4f})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, color="steelblue", alpha=0.7, s=30, zorder=3)

    slope, intercept, r, p, se = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="firebrick", linewidth=2,
            label=f"linear fit: r={r:+.3f}, R²={r**2:.3f}, p={p:.4f}")
    ax.legend(loc="best", fontsize=9)

    ax.set_xlabel("Distance from plate centroid (pixels)", fontsize=11)
    ax.set_ylabel(f"{args.metric} — Active pixels (A.U.)", fontsize=11)
    ax.set_title(f"{args.metric} vs. Distance from Plate Centroid  (n={len(x)})", fontsize=12)
    plt.tight_layout()

    save_path = None
    if args.save:
        stem = os.path.splitext(os.path.basename(args.results))[0]
        stem_dir = os.path.dirname(os.path.abspath(args.results))
        save_path = os.path.join(stem_dir, f"{stem}_{args.metric}_centroid_distance.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Position] Saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()

