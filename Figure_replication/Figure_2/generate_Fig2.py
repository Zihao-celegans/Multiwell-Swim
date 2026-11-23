"""Figure 2: gallery of a representative well at a given dose across timepoints.

Frames are pulled from the raw .avi recordings (nearest video to each requested
elapsed time, measured from the first video's timestamp) and cropped to the well
using the ROI info saved by the pipeline (full-resolution pixel coordinates).

Example:
    python generate_Fig2.py --dose "0 mM" --times 0 60 240 480

Each panel is built in two steps. (1) The intensity-change map (T0 - T1, where
T1 = T0 + dt, default 1 s) is denoised by zeroing changes below +/-thresh, then
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
    p.add_argument("--dose", default="0 mM", help="Condition label as written in the layout CSV")
    p.add_argument("--well", default=None, help="Well label (e.g. C3); default: first well of the dose")
    p.add_argument("--times", type=float, nargs="+", default=[0, 60, 240, 480], help="Elapsed minutes")
    p.add_argument("--frame", type=int, default=0, help="Frame index (T0) within each video")
    p.add_argument("--dt", type=float, default=1.0, help="Seconds between the T0 and T1 frames")
    p.add_argument("--sat", type=float, default=50,
                   help="Intensity change (gray levels) at which the color saturates to pure red/blue")
    p.add_argument("--thresh", type=float, default=15,
                   help="Noise level: |change| (gray levels) below this is set to 0 (black) in the difference map")
    p.add_argument("--gain", type=float, default=1.0, help="merged view: brightness multiplier for the raw base image (1 = as acquired)")
    p.add_argument("--view", choices=["merged", "map", "raw"], default="merged",
                   help="merged: difference map blended into the raw image; map: color-coded "
                        "intensity-change map only; raw: single T0 grayscale frame")
    p.add_argument("--pad", type=float, default=1.0, help="Crop half-width as a multiple of the ROI radius")
    p.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "Fig2_gallery.png"))
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    layout = load_layout(args.layout)
    wells = layout[args.dose]
    well = args.well or wells[0]
    if well not in wells:
        raise SystemExit(f"Well {well} is not assigned to {args.dose}; options: {wells}")

    videos = list_videos(args.data_dir)
    panels = []
    for t in args.times:
        elapsed, avi, roi = min(videos, key=lambda v: abs(v[0] - t))
        print(f"{t:g} min -> {os.path.basename(avi)} (actual {elapsed:.1f} min), well {well}")
        cap = cv2.VideoCapture(avi)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        frame_t1 = args.frame + int(round(args.dt * fps))
        panels.append(crop_well_frames(avi, roi, well, [args.frame, frame_t1], args.pad))

    # Step 1: denoised intensity-change maps, one shared scale for all panels.
    # Changes below +/-thresh are zeroed (black); colors saturate at +/-sat. The
    # color bar spans +/-(largest |change|), so the saturated plateaus beyond +/-sat are visible.
    diffs = [denoise_diff(a, b, args.thresh) for a, b in panels]
    # Symmetric bar: +/-(largest |change| in any panel); values beyond the data
    # range on the weaker side simply stay fully saturated.
    lim = max(float(np.abs(d).max()) for d in diffs)
    dmin, dmax = -lim, lim
    print(f"True difference range across panels: {min(float(d.min()) for d in diffs):g} to "
          f"{max(float(d.max()) for d in diffs):g}; color bar: +/-{lim:g}")
    bar_cmap = ListedColormap(change_to_rgb(np.linspace(dmin, dmax, 512), args.thresh, args.sat))
    ticks = [dmin, 0.0, dmax]

    fig, axes = plt.subplots(1, len(args.times), figsize=(3 * len(args.times), 3.2))
    for ax, t, (img0, img1), diff in zip(axes, args.times, panels, diffs):
        if args.view == "raw":
            ax.imshow(img0, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        else:
            diff_rgb = change_to_rgb(diff, args.thresh, args.sat)
            # Step 2 (merged view): blend the map into the raw image.
            ax.imshow(diff_rgb if args.view == "map" else merge_map(img0, img1, diff_rgb, args.gain),
                      interpolation="nearest")
        ax.set_title(f"{t:g} min", fontsize=12)
        ax.axis("off")

    fig.suptitle(f"{args.dose} (well {well})", fontsize=12)
    if args.view == "raw":
        fig.tight_layout()
    else:
        fig.tight_layout(rect=[0.09, 0, 1, 1])
        cax = fig.add_axes([0.065, 0.2, 0.012, 0.55])
        cbar = fig.colorbar(ScalarMappable(norm=Normalize(dmin, dmax), cmap=bar_cmap), cax=cax)
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
        cbar.set_label("Intensity change", fontsize=12, fontfamily="Arial")
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{v:g}" for v in ticks])
        cbar.ax.tick_params(labelsize=10, length=0)
        cbar.outline.set_visible(False)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontfamily("Arial")
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
