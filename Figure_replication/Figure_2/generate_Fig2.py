"""Figure 2: gallery of a representative well at a given dose across timepoints.

Frames are pulled from the raw .avi recordings (nearest video to each requested
elapsed time, measured from the first video's timestamp) and cropped to the well
using the ROI info saved by the pipeline (full-resolution pixel coordinates).

Example:
    python generate_Fig2.py --doses "0 mM" "0.0125 mM" "0.025 mM" "0.1 mM" --times 0 90 240 480

Rows are doses, columns are timepoints. For each dose the well whose activity
(--metric, default ActValS) at --select_time (default 90 min) is closest to the
mean of all wells at that dose is shown. Alternatives: --select center (well
closest to the plate center) or --wells to choose explicitly.

Each panel is built in two steps. (1) The intensity-change map (T0 - T1, where
T1 = T0 + dt, default 3 s) is denoised by zeroing changes below +/-thresh, then
color-coded: red where the T0 frame is brighter, blue where the later frame is
brighter, black for no (or sub-noise) change. (2) The map is blended into the raw
image, using the map's brightness as opacity, so no-change pixels show the raw
grayscale. The color bar applies to the intensity-change map only. Use
--view map for the difference map alone.
"""

import argparse
import csv
import glob
import json
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

ROW_LABELS = "ABCDEFGH"

DEFAULT_DATA_DIR = r"D:\MultiWell_swim\Preliminary\08032026_N2_Levamisole_DRC_test02"
DEFAULT_LAYOUT = (
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\Multiwell_swim"
    r"\Well_assignments\Well_assignment_0728\dose_layout.csv"
)
TIMESTAMP_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})")


def load_layout(path: str) -> dict:
    """Return {condition: [well, ...]} from a condition,well CSV."""
    layout = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            layout.setdefault(row["condition"].strip(), []).append(row["well"].strip())
    return layout


def list_videos(data_dir: str) -> list:
    """Return [(elapsed_min, avi_path, roi_path)] sorted chronologically."""
    entries = []
    prev, offset = None, 0
    for avi in sorted(glob.glob(os.path.join(data_dir, "*.avi"))):
        stem = os.path.splitext(os.path.basename(avi))[0]
        hh, mm, ss = (int(g) for g in TIMESTAMP_RE.search(stem).groups())
        sec = hh * 3600 + mm * 60 + ss
        if prev is not None and sec < prev:  # midnight rollover
            offset += 24 * 3600
        prev = sec
        roi = os.path.join(data_dir, f"{stem}_output", f"{stem}_roi_info.json")
        entries.append((sec + offset, avi, roi))
    t0 = entries[0][0]
    return [((s - t0) / 60.0, avi, roi) for s, avi, roi in entries]


def well_index(well: str, num_col: int) -> int:
    return ROW_LABELS.index(well[0]) * num_col + int(well[1:]) - 1


def closest_to_center(wells: list, roi_info: dict) -> str:
    """Well (from `wells`) whose ROI center is nearest the plate center.

    Plate center = mean of all ROI centers; ties go to the first in `wells`.
    """
    centers = np.array([r["center"] for r in roi_info["roi"]])
    plate_center = centers.mean(axis=0)
    dist = lambda w: float(np.hypot(*(centers[well_index(w, roi_info["num_col"])] - plate_center)))
    return min(wells, key=dist)


def closest_to_mean_activity(wells: list, results_path: str, metric: str) -> tuple:
    """Well (from `wells`) whose activity is closest to the mean activity of `wells`.

    Activity comes from one video's '<stem>_results.json' (per-well matrix under
    `metric`); wells with NaN activity are ignored. Returns (well, mean, value).
    """
    with open(results_path) as fh:
        matrix = np.array(json.load(fh)[metric], dtype=np.float64)
    num_col = matrix.shape[1]
    values = {w: matrix.flat[well_index(w, num_col)] for w in wells}
    values = {w: v for w, v in values.items() if not np.isnan(v)}
    mean = float(np.mean(list(values.values())))
    best = min(values, key=lambda w: abs(values[w] - mean))
    return best, mean, values[best]


def crop_well_frames(avi: str, roi_path: str, well: str, frame_idxs: list, pad: float) -> list:
    """Return grayscale (uint8) crops of the well for each requested frame index."""
    with open(roi_path) as fh:
        roi = json.load(fh)
    entry = roi["roi"][well_index(well, roi["num_col"])]
    cx, cy = entry["center"]
    r = entry["radius"] * pad

    cap = cv2.VideoCapture(avi)
    crops = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Could not read frame {idx} from {avi}")
        gray = frame[:, :, :3].mean(axis=2) if frame.ndim == 3 else frame
        h, w = gray.shape
        x0, x1 = max(int(cx - r), 0), min(int(cx + r), w)
        y0, y1 = max(int(cy - r), 0), min(int(cy + r), h)
        crops.append(gray[y0:y1, x0:x1])
    cap.release()
    return crops


def denoise_diff(img_t0, img_t1, thresh: float):
    """Intensity change T0 - T1 (gray levels) with noise removed.

    Changes smaller than `thresh` in magnitude are set to 0 (no change).
    """
    diff = img_t0.astype(np.float32) - img_t1.astype(np.float32)
    diff[np.abs(diff) < thresh] = 0
    return diff


def change_to_rgb(diff, thresh: float, sat: float):
    """Map intensity change to RGB (float, [0, 1]): blue <- black -> red.

    Black for |change| <= thresh (including the 0-change noise plateau), then
    brightening linearly to pure red (T0 brighter) / pure blue (later frame
    brighter) at |change| = sat; beyond sat the color stays fully saturated.
    Works on arrays and is used for both the image and the color bar.
    """
    diff = np.asarray(diff, dtype=np.float32)
    level = np.clip((np.abs(diff) - thresh) / max(sat - thresh, 1e-6), 0, 1)
    rgb = np.zeros(diff.shape + (3,), dtype=np.float32)
    rgb[..., 0] = np.where(diff > 0, level, 0)
    rgb[..., 2] = np.where(diff < 0, level, 0)
    return rgb


def merge_map(img_t0, img_t1, diff_rgb, gain: float):
    """Blend the (already denoised) intensity-change map into the raw image.

    Base is the grayscale mean of the two frames. The map's own brightness is
    the per-pixel opacity: black (no change) leaves the raw image untouched and
    saturated red/blue fully replaces it.
    """
    a = img_t0.astype(np.float32)
    b = img_t1.astype(np.float32)
    gray = np.clip((a + b) / 2 * gain / 255.0, 0, 1)
    base = np.repeat(gray[..., None], 3, axis=2)
    alpha = diff_rgb.max(axis=2, keepdims=True)
    return np.clip(base * (1 - alpha) + diff_rgb, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--layout", default=DEFAULT_LAYOUT)
    p.add_argument("--doses", nargs="+", default=["0 mM", "0.0125 mM", "0.025 mM", "0.1 mM"],
                   help="Condition labels as written in the layout CSV; one gallery row per dose")
    p.add_argument("--wells", nargs="+", default=None,
                   help="Well label per dose (same order as --doses); overrides --select")
    p.add_argument("--select", choices=["mean-activity", "center"], default="mean-activity",
                   help="Well selection per dose: mean-activity = well whose activity at --select_time is closest "
                        "to the dose's mean; center = well closest to the plate center")
    p.add_argument("--select_time", type=float, default=90, help="Elapsed minutes at which activity is compared")
    p.add_argument("--metric", choices=["ActVal", "ActValS"], default="ActValS",
                   help="Per-well activity metric from the results JSON used by --select mean-activity")
    p.add_argument("--times", type=float, nargs="+", default=[0, 90, 240, 480], help="Elapsed minutes")
    p.add_argument("--frame", type=int, default=0, help="Frame index (T0) within each video")
    p.add_argument("--dt", type=float, default=3.0, help="Seconds between the T0 and T1 frames")
    p.add_argument("--sat", type=float, default=40,
                   help="Intensity change (gray levels) at which the color saturates to pure red/blue")
    p.add_argument("--thresh", type=float, default=10,
                   help="Noise level: |change| (gray levels) below this is set to 0 (black) in the difference map")
    p.add_argument("--gain", type=float, default=1.0, help="merged view: brightness multiplier for the raw base image (1 = as acquired)")
    p.add_argument("--view", choices=["merged", "map", "raw"], default="merged",
                   help="merged: difference map blended into the raw image; map: color-coded "
                        "intensity-change map only; raw: single T0 grayscale frame")
    p.add_argument("--pad", type=float, default=1.0, help="Crop half-width as a multiple of the ROI radius")
    p.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "Fig2_gallery.svg"),
                   help="Output path; the format follows the extension (.svg, .png, .pdf)")
    p.add_argument("--size", type=float, nargs="+", default=[3.5], metavar="INCHES",
                   help="Figure size in inches: 'W' (height fitted to the content, default) or 'W H'")
    p.add_argument("--margins", type=float, nargs=5, default=[0.36, 0.5, 0.1, 0.34, 0.03],
                   metavar=("LEFT", "RIGHT", "TOP", "BOTTOM", "GAP"),
                   help="Inches: figure margins (right holds the color bar, bottom the time axis) "
                        "and the gap between panels")
    p.add_argument("--bar_offset", type=float, default=0.05, help="Inches from the panel grid to the color bar")
    p.add_argument("--axis_label_offset", type=float, default=0.27,
                   help="Inches from the panel grid's left edge to the 'Dose' axis label")
    p.add_argument("--axis_label_offset_bottom", type=float, default=0.25,
                   help="Inches from the panel grid's bottom edge to the 'Time' axis label")
    p.add_argument("--dpi", type=int, default=300, help="Resolution for raster output formats")
    p.add_argument("--also_png", action="store_true",
                   help="Also save a PNG next to --output (same name, .png, at --dpi)")
    args = p.parse_args()

    # All text: Arial 10 pt (titles included; matplotlib's default title size is larger).
    # svg.fonttype=none keeps text as editable <text> (font-family Arial) rather than outlines.
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.titlesize": 10,
                         "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "svg.fonttype": "none"})

    layout = load_layout(args.layout)
    videos = list_videos(args.data_dir)
    with open(videos[0][2]) as fh:
        roi0 = json.load(fh)
    if args.wells:
        wells = args.wells
    elif args.select == "center":
        wells = [closest_to_center(layout[d], roi0) for d in args.doses]
    else:
        sel_min, sel_avi, sel_roi = min(videos, key=lambda v: abs(v[0] - args.select_time))
        sel_results = sel_roi.replace("_roi_info.json", "_results.json")
        print(f"Well selection: {args.metric} at {args.select_time:g} min "
              f"({os.path.basename(sel_avi)}, actual {sel_min:.1f} min)")
        wells = []
        for d in args.doses:
            well, mean, value = closest_to_mean_activity(layout[d], sel_results, args.metric)
            print(f"  {d}: mean {mean:.1f}, selected {well} ({value:.1f})")
            wells.append(well)
    if len(wells) != len(args.doses):
        raise SystemExit("--wells must have one well per dose")
    for dose, well in zip(args.doses, wells):
        if well not in layout[dose]:
            raise SystemExit(f"Well {well} is not assigned to {dose}; options: {layout[dose]}")
        print(f"{dose}: well {well}")

    # grid[row = dose][col = time] = (frame at T0, frame at T1)
    grid = []
    for dose, well in zip(args.doses, wells):
        row = []
        for t in args.times:
            elapsed, avi, roi = min(videos, key=lambda v: abs(v[0] - t))
            print(f"{dose} / {well}: {t:g} min -> {os.path.basename(avi)} (actual {elapsed:.1f} min)")
            cap = cv2.VideoCapture(avi)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            frame_t1 = args.frame + int(round(args.dt * fps))
            row.append(crop_well_frames(avi, roi, well, [args.frame, frame_t1], args.pad))
        grid.append(row)

    # Step 1: denoised intensity-change maps, one shared scale for all panels.
    # Changes below +/-thresh are zeroed (black); colors saturate at +/-sat. The
    # color bar spans +/-(largest |change|), so the saturated plateaus beyond +/-sat are visible.
    diffs = [[denoise_diff(a, b, args.thresh) for a, b in row] for row in grid]
    flat = [d for row in diffs for d in row]
    lim = max(float(np.abs(d).max()) for d in flat)
    dmin, dmax = -lim, lim
    print(f"True difference range across panels: {min(float(d.min()) for d in flat):g} to "
          f"{max(float(d.max()) for d in flat):g}; color bar: +/-{lim:g}")
    bar_cmap = ListedColormap(change_to_rgb(np.linspace(dmin, dmax, 512), args.thresh, args.sat))
    ticks = [dmin, 0.0, dmax]

    nrow, ncol = len(args.doses), len(args.times)
    fig_w, fig_h = (args.size + [None])[:2]
    # Layout in inches: square panels, top-aligned. The left margin holds the dose
    # tick labels + axis label, the bottom margin the time tick labels + axis
    # label, and the right margin the color bar. Panel size is set by the width;
    # if no height is given, the figure height is fitted to the content.
    left, right, top, bottom, gap = args.margins
    s = (fig_w - left - right - (ncol - 1) * gap) / ncol
    if fig_h is None:
        fig_h = top + bottom + nrow * s + (nrow - 1) * gap
    else:
        s = min(s, (fig_h - top - bottom - (nrow - 1) * gap) / nrow)
    fig = plt.figure(figsize=(fig_w, fig_h))
    grid_w = ncol * s + (ncol - 1) * gap
    grid_h = nrow * s + (nrow - 1) * gap
    axes = []
    for i, (dose, well) in enumerate(zip(args.doses, wells)):
        row = []
        for j, t in enumerate(args.times):
            x0 = left + j * (s + gap)
            y0 = fig_h - top - (i + 1) * s - i * gap
            ax = fig.add_axes([x0 / fig_w, y0 / fig_h, s / fig_w, s / fig_h])
            row.append(ax)
            img0, img1 = grid[i][j]
            if args.view == "raw":
                ax.imshow(img0, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            else:
                diff_rgb = change_to_rgb(diffs[i][j], args.thresh, args.sat)
                # Step 2 (merged view): blend the map into the raw image.
                ax.imshow(diff_rgb if args.view == "map" else merge_map(img0, img1, diff_rgb, args.gain),
                          interpolation="nearest")
            ax.axis("off")
            if i == nrow - 1:
                ax.annotate(f"{t:g}", (0.5, 0), xytext=(0, -2), textcoords="offset points",
                            xycoords="axes fraction", ha="center", va="top")
            if j == 0:
                ax.text(-0.04, 0.5, dose.rsplit(" ", 1)[0], transform=ax.transAxes, rotation=90,
                        ha="right", va="center")
        axes.append(row)

    # Axis labels: units live here rather than on every tick label.
    dose_unit = args.doses[0].rsplit(" ", 1)[-1]
    fig.text((left - args.axis_label_offset) / fig_w, 1 - (top + grid_h / 2) / fig_h, f"LEV dose ({dose_unit})",
             rotation=90, ha="center", va="center")
    fig.text((left + grid_w / 2) / fig_w, (fig_h - top - grid_h - args.axis_label_offset_bottom) / fig_h,
             "Time (min)", ha="center", va="center")

    if args.view != "raw":
        # Color bar on the right, spanning the full height of the panel grid.
        cax = fig.add_axes([(left + grid_w + args.bar_offset) / fig_w, (fig_h - top - grid_h) / fig_h,
                            0.07 / fig_w, grid_h / fig_h])
        cbar = fig.colorbar(ScalarMappable(norm=Normalize(dmin, dmax), cmap=bar_cmap), cax=cax)
        cbar.set_label("Intensity change", labelpad=0)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{v:g}" for v in ticks])
        cbar.ax.tick_params(length=0, pad=1)
        cbar.outline.set_visible(False)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")
    if args.also_png:
        png_path = os.path.splitext(args.output)[0] + ".png"
        fig.savefig(png_path, dpi=args.dpi)
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
