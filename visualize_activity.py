"""
visualize_activity.py — Visualize activity results from activity_analysis.py.

Produces three figures from a single results JSON file:

  Figure 1 — Plate heatmap
    ActVal and ActValS displayed as colour-coded 8×12 grids, with well
    row/column labels.  Mirrors the intuition of MATLAB imagesc on the
    reshaped ActVal matrix.

  Figure 2 — Per-well activity overlay on the first video frame
    Each ROI circle is colour-coded by ActVal (cold→hot colourmap).
    Gives a spatial sense of which wells have active worms.

  Figure 3 — Distribution (strip + box plot)
    All 96 ActVal values shown as individual points with a box overlay,
    so outlier wells are immediately visible.

Usage:
    python visualize_activity.py --results PATH_TO_results.json
                                 [--video  PATH_TO_VIDEO]   # for Figure 2
                                 [--roi    PATH_TO_roi_info.json]  # for Figure 2
                                 [--save]                   # save PNGs alongside results
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _load_roi(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _read_first_frame(video_path: str) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read first frame from: {video_path}")
    if frame.ndim == 3:
        return frame[:, :, :3].mean(axis=2).astype(np.float32)
    return frame.astype(np.float32)


# ---------------------------------------------------------------------------
# Figure 1 — Plate heatmap
# ---------------------------------------------------------------------------

def plot_plate_heatmap(act_val: np.ndarray, act_val_s: np.ndarray,
                       title_prefix: str, save_path: str = None) -> None:
    """Colour-coded 8×12 plate grid for ActVal and ActValS side-by-side."""
    num_row, num_col = act_val.shape

    row_labels = [chr(ord("A") + r) for r in range(num_row)]   # A–H
    col_labels = [str(c + 1) for c in range(num_col)]           # 1–12

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, label in zip(axes,
                                [act_val, act_val_s],
                                ["ActVal  (adjacent frames)",
                                 "ActValS  (frame-skip)"]):
        im = ax.imshow(data, cmap="inferno", aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(np.arange(num_col))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(np.arange(num_row))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(label, fontsize=11)

        # Annotate each cell with its numeric value
        vmax = np.nanmax(data)
        for r in range(num_row):
            for c in range(num_col):
                val = data[r, c]
                if not np.isnan(val):
                    # White text on dark cells, black on bright cells
                    brightness = val / vmax if vmax > 0 else 0
                    txt_color = "white" if brightness < 0.6 else "black"
                    ax.text(c, r, f"{val:.0f}", ha="center", va="center",
                            fontsize=5, color=txt_color)

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                     label="Active pixels (A.U.)")

    fig.suptitle(f"{title_prefix} — Plate Activity Heatmap", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 2 — Spatial overlay on first video frame
# ---------------------------------------------------------------------------

def plot_spatial_overlay(frame: np.ndarray, roi_list: list,
                         act_val_flat: np.ndarray,
                         title_prefix: str, save_path: str = None) -> None:
    """Draw ROI circles on the video frame, colour-coded by ActVal."""
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(frame, cmap="gray", aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    # Normalise ActVal to [0, 1] for colour mapping
    vmin = np.nanmin(act_val_flat)
    vmax = np.nanmax(act_val_flat)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("inferno")

    for roi, val in zip(roi_list, act_val_flat):
        cx, cy = roi["center"]
        r = roi["radius"]
        color = cmap(norm(val)) if not np.isnan(val) else (0.3, 0.3, 0.3, 1.0)
        circle = plt.Circle((cx, cy), r, color=color, fill=False, linewidth=1.5)
        ax.add_patch(circle)

    # Colourbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01,
                 label="ActVal — Active pixels (A.U.)")

    ax.set_title(f"{title_prefix} — Activity Spatial Overlay", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 3 — Distribution plot
# ---------------------------------------------------------------------------

def plot_distribution(act_val_flat: np.ndarray, act_val_s_flat: np.ndarray,
                      title_prefix: str, save_path: str = None) -> None:
    """Strip + box plot of per-well activity values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    rng = np.random.default_rng(0)   # reproducible jitter

    for ax, data, label in zip(axes,
                                [act_val_flat, act_val_s_flat],
                                ["ActVal  (adjacent frames)",
                                 "ActValS  (frame-skip)"]):
        valid = data[~np.isnan(data)]

        # Box plot
        bp = ax.boxplot(valid, widths=0.4, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        boxprops=dict(facecolor="#444444", alpha=0.6),
                        whiskerprops=dict(color="#444444"),
                        capprops=dict(color="#444444"),
                        flierprops=dict(marker=""))

        # Strip (jittered individual points)
        jitter = rng.uniform(-0.15, 0.15, size=len(valid))
        ax.scatter(1 + jitter, valid, color="steelblue",
                   s=20, alpha=0.7, zorder=3)

        ax.set_xticks([1])
        ax.set_xticklabels([label], fontsize=9)
        ax.set_ylabel("Active pixels (A.U.)", fontsize=9)

        # Annotate median and n
        median_val = float(np.median(valid))
        ax.text(1.25, median_val,
                f"median={median_val:.1f}\nn={len(valid)}",
                va="center", fontsize=8, color="dimgray")

    fig.suptitle(f"{title_prefix} — Per-well Activity Distribution", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 4 — Timecourse heatmap (kymograph)
# ---------------------------------------------------------------------------

def plot_timecourse_heatmap(act_val_s_tc: np.ndarray, fps: float,
                            num_row: int, num_col: int,
                            title_prefix: str, save_path: str = None) -> None:
    """Kymograph: x = time (s), y = well index, colour = ActValS.

    Wells are ordered row-major (A1, A2, …, A12, B1, …, H12) on the y-axis,
    with row-group dividers and A–H row labels to aid plate orientation.
    Mirrors the MATLAB imagesc(T, G, ActAll) kymograph in VisualData_5HT.m.

    Parameters
    ----------
    act_val_s_tc : np.ndarray, shape (num_roi, num_frames)
        Full per-frame ActValS timecourse as returned by activity_analysis.
    fps          : frame rate in Hz (used to convert frame index → seconds).
    """
    num_roi, num_frames = act_val_s_tc.shape

    # Time axis: frame transition t (0-based) → time in seconds
    # Transition t is computed between frame t and frame t+1, so its
    # representative time is (t + 0.5) / fps.
    t_axis = (np.arange(num_frames) + 0.5) / fps          # seconds

    # Well axis: 0-based index, one entry per ROI
    well_idx = np.arange(num_roi)

    fig, ax = plt.subplots(figsize=(14, max(6, num_roi // 8)))

    im = ax.imshow(
        act_val_s_tc,
        aspect="auto",
        cmap="inferno",
        interpolation="nearest",
        origin="upper",
        extent=[t_axis[0], t_axis[-1], num_roi - 0.5, -0.5],
    )

    # Row-group dividers and A–H labels (one group = num_col wells)
    row_labels = [chr(ord("A") + r) for r in range(num_row)]
    for r in range(num_row):
        y_mid = r * num_col + num_col / 2 - 0.5    # centre of the row group
        ax.text(-t_axis[-1] * 0.01, y_mid, row_labels[r],
                ha="right", va="center", fontsize=9, color="white",
                fontweight="bold")
        if r > 0:
            ax.axhline(r * num_col - 0.5, color="white", linewidth=0.5,
                       linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Well (row-major: A1 … H12)", fontsize=11)
    ax.set_title(f"{title_prefix} — ActValS Timecourse", fontsize=12)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 label="ActValS — Active pixels (A.U.)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize activity results from activity_analysis.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True,
                        help="Path to the _results.json file.")
    parser.add_argument("--video", default=None,
                        help="Path to the original video (needed for Figure 2).")
    parser.add_argument("--roi", default=None,
                        help="Path to roi_info.json (needed for Figure 2).")
    parser.add_argument("--save", action="store_true",
                        help="Save each figure as a PNG alongside the results file.")
    args = parser.parse_args()

    # ── Load results ──────────────────────────────────────────────────────────
    print(f"[Viz] Loading results from: {args.results}")
    res = _load_results(args.results)

    act_val   = np.array(res["ActVal"],  dtype=np.float64)   # (num_row, num_col)
    act_val_s = np.array(res["ActValS"], dtype=np.float64)
    vidname   = res.get("Vidname", os.path.basename(args.results))
    title_prefix = os.path.splitext(vidname)[0]

    stem = os.path.join(
        os.path.dirname(os.path.abspath(args.results)),
        os.path.splitext(os.path.basename(args.results))[0],
    )

    # ── Figure 1: Plate heatmap ───────────────────────────────────────────────
    save1 = stem + "_heatmap.png" if args.save else None
    plot_plate_heatmap(act_val, act_val_s, title_prefix, save_path=save1)

    # ── Figure 2: Spatial overlay (requires --video and --roi) ───────────────
    if args.video and args.roi:
        frame    = _read_first_frame(args.video)
        roi_data = _load_roi(args.roi)
        roi_list = roi_data["roi"]
        act_flat = act_val.flatten()          # row-major → same order as roi_list
        save2 = stem + "_overlay.png" if args.save else None
        plot_spatial_overlay(frame, roi_list, act_flat, title_prefix, save_path=save2)
    else:
        print("[Viz] Skipping Figure 2 (spatial overlay) — provide --video and --roi.")

    # ── Figure 3: Distribution ────────────────────────────────────────────────
    save3 = stem + "_distribution.png" if args.save else None
    plot_distribution(act_val.flatten(), act_val_s.flatten(),
                      title_prefix, save_path=save3)

    # ── Figure 4: Timecourse heatmap ─────────────────────────────────────────
    if "ActValS_timecourse" in res and res["ActValS_timecourse"] is not None:
        tc_raw = res["ActValS_timecourse"]   # list of lists, None = NaN
        tc = np.array(
            [[np.nan if v is None else v for v in row] for row in tc_raw],
            dtype=np.float64,
        )  # (num_roi, num_frames)
        fps = float(res.get("FrameRate", 5.0))
        num_row = len(act_val)         # act_val is (num_row, num_col)
        num_col = len(act_val[0])
        save4 = stem + "_timecourse.png" if args.save else None
        plot_timecourse_heatmap(tc, fps, num_row, num_col,
                                title_prefix, save_path=save4)
    else:
        print("[Viz] Skipping Figure 4 (timecourse) — "
              "re-run activity_analysis.py to generate ActValS_timecourse.")

    print("[Viz] Done.")


if __name__ == "__main__":
    main()
