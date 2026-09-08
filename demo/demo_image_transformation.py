"""
demo_image_transformation.py — Generate presentation images showing each
intermediate step of the Activity Analysis image-processing pipeline
(see activity_analysis.py), computed on one frame pair from a sample video.

Read-only: frames are read from the video via OpenCV (VideoCapture, seek +
read); nothing is ever written back to the video file or its folder. Outputs
go to <video_stem>_demo/ under the Dropbox presentations folder by default
(never into the raw-data folder) — override with --output_dir.

Pipeline mirrored (see activity_analysis.py for the full math):
    1. Raw frame A, raw frame B (grayscale)
    2. Normalised diff:        |A - B| / (A + B)
    3. Gaussian-smoothed diff: convolve2d(diff, gau, 'same')
    4. Thresholded binary:     smoothed > noise_threshold

Optionally, pass --roi and --well (e.g. "D6") to also crop every step down
to a single well, using the exact same edge-safe crop the real analysis uses
per-ROI (activity_analysis._roi_activity): the diff is cropped first, then
convolved — not the other way around — so the well-level images match what
the pipeline actually measures for that well.

Usage:
    python demo/demo_image_transformation.py \\
        --video "E:/MultiWell_swim/09062026_CeDiv_Pyrantel_test01/control/video0000 11-16-11.avi"

    # Custom frame pair / parameters, custom output location
    python demo/demo_image_transformation.py --video "..." \\
        --frame_a 0 --frame_b 5 --noise_threshold 0.6 --gaussian_std 0.9 \\
        --output_dir "C:/Users/.../presentation_demo"

    # Also crop every step down to well D6
    python demo/demo_image_transformation.py --video "..." \\
        --roi "E:/.../video0000 11-16-11_output/video0000 11-16-11_roi_info.json" \\
        --well D6
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# activity_analysis.py and utils.py live in the repo root, one level up from
# this demo/ folder — add it to sys.path so these imports resolve regardless
# of the current working directory the script is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activity_analysis import build_gaussian_kernel, _normalised_diff
from utils import load_roi, probe_video, read_video_frame

DEFAULT_OUTPUT_ROOT = r"C:\Users\jl200\Dropbox\Weekly_Updates_JHU\Lab_meetings\Presentations\Swim_0909"


def _save_gray(img: np.ndarray, path: str, vmin: float, vmax: float) -> None:
    """Save a single-channel array as a clean grayscale PNG (no axes/border)."""
    plt.imsave(path, img, cmap="gray", vmin=vmin, vmax=vmax)


def _well_label_to_index(well: str, num_row: int, num_col: int) -> int:
    """Convert a plate label like "D6" to a row-major ROI list index."""
    well = well.strip().upper()
    row = ord(well[0]) - ord("A")
    col = int(well[1:]) - 1
    if not (0 <= row < num_row) or not (0 <= col < num_col):
        raise ValueError(
            f"Well '{well}' is out of range for a {num_row}x{num_col} plate "
            f"(rows A-{chr(ord('A') + num_row - 1)}, cols 1-{num_col})."
        )
    return row * num_col + col


def _crop_square(img: np.ndarray, xc: int, yc: int, radius: int) -> np.ndarray:
    """Zero-padded (2R+1)x(2R+1) crop centred at (xc, yc).

    Mirrors the edge-safe crop in activity_analysis._roi_activity so wells
    near the frame boundary are handled identically here.
    """
    img_h, img_w = img.shape
    size = 2 * radius + 1
    y0, y1 = yc - radius, yc + radius + 1
    x0, x1 = xc - radius, xc + radius + 1
    y0_clip, y1_clip = max(y0, 0), min(y1, img_h)
    x0_clip, x1_clip = max(x0, 0), min(x1, img_w)

    crop = np.zeros((size, size), dtype=img.dtype)
    if y0_clip < y1_clip and x0_clip < x1_clip:
        py0, px0 = y0_clip - y0, x0_clip - x0
        py1, px1 = py0 + (y1_clip - y0_clip), px0 + (x1_clip - x0_clip)
        crop[py0:py1, px0:px1] = img[y0_clip:y1_clip, x0_clip:x1_clip]
    return crop


def generate(video_path: str, output_dir: str, frame_a_idx: int, frame_b_idx: int,
             noise_threshold: float, gaussian_std: float,
             roi_path: str = None, well: str = None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Demo] Reading frames {frame_a_idx} and {frame_b_idx} from: {video_path}")
    frame_a = read_video_frame(video_path, frame_a_idx)   # float32 grayscale, read-only seek
    frame_b = read_video_frame(video_path, frame_b_idx)

    # ── Step 1: raw frames ───────────────────────────────────────────────────
    _save_gray(frame_a, os.path.join(output_dir, "1_frame_a_raw.png"), 0, 255)
    _save_gray(frame_b, os.path.join(output_dir, "2_frame_b_raw.png"), 0, 255)

    # ── Step 2: normalised diff  |A-B|/(A+B) ─────────────────────────────────
    diff = _normalised_diff(frame_a.astype(np.float64), frame_b.astype(np.float64))
    diff_clean = np.nan_to_num(diff, nan=0.0)
    _save_gray(diff_clean, os.path.join(output_dir, "3_normalised_diff.png"), 0, 0.7)

    # ── Step 3: Gaussian-smoothed diff ───────────────────────────────────────
    gau = build_gaussian_kernel(gaussian_std)
    smoothed = convolve2d(diff_clean, gau, mode="same", boundary="fill", fillvalue=0.0)
    _save_gray(smoothed, os.path.join(output_dir, "4_smoothed_diff.png"), 0, smoothed.max())

    # ── Step 4: thresholded binary mask ──────────────────────────────────────
    binary = (smoothed > noise_threshold).astype(np.float64)
    _save_gray(binary, os.path.join(output_dir, "5_binary_threshold.png"), 0, 1)

    # ── Combined annotated overview (single presentation slide) ─────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    panels = [
        (frame_a, "1. Frame A (raw)", 0, 255),
        (frame_b, "2. Frame B (raw)", 0, 255),
        (diff_clean, "3. Normalised diff\n|A-B|/(A+B)", 0, 0.7),
        (smoothed, "4. Gaussian-smoothed", 0, smoothed.max()),
        (binary, f"5. Thresholded\n(> {noise_threshold})", 0, 1),
    ]
    for ax, (img, title, vmin, vmax) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.suptitle(
        f"Activity Analysis — Image Transformation Pipeline "
        f"(frames {frame_a_idx} \u2192 {frame_b_idx})",
        fontsize=13, y=1.05,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "0_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Demo] Saved 6 images to: {output_dir}")

    # ── Optional: crop every step down to a single well ──────────────────────
    if roi_path and well:
        roi_data = load_roi(roi_path)
        roi_list = roi_data["roi"]
        num_row, num_col = roi_data["num_row"], roi_data["num_col"]
        well_idx = _well_label_to_index(well, num_row, num_col)
        roi = roi_list[well_idx]
        xc = int(round(roi["center"][0]))
        yc = int(round(roi["center"][1]))
        radius = int(round(roi["radius"]))

        well_dir = os.path.join(output_dir, f"well_{well.upper()}")
        os.makedirs(well_dir, exist_ok=True)

        well_frame_a = _crop_square(frame_a, xc, yc, radius)
        well_frame_b = _crop_square(frame_b, xc, yc, radius)
        # Diff is computed on the full frame first, then cropped — matching
        # activity_analysis.py (the well only ever sees the global diff crop,
        # never a diff of pre-cropped raw frames).
        well_diff = _crop_square(diff_clean, xc, yc, radius)
        well_smoothed = convolve2d(well_diff, gau, mode="same", boundary="fill", fillvalue=0.0)
        well_binary = (well_smoothed > noise_threshold).astype(np.float64)

        _save_gray(well_frame_a, os.path.join(well_dir, "1_frame_a_raw.png"), 0, 255)
        _save_gray(well_frame_b, os.path.join(well_dir, "2_frame_b_raw.png"), 0, 255)
        _save_gray(well_diff, os.path.join(well_dir, "3_normalised_diff.png"), 0, 0.7)
        _save_gray(well_smoothed, os.path.join(well_dir, "4_smoothed_diff.png"), 0, well_smoothed.max())
        _save_gray(well_binary, os.path.join(well_dir, "5_binary_threshold.png"), 0, 1)

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        panels = [
            (well_frame_a, "1. Frame A (raw)", 0, 255),
            (well_frame_b, "2. Frame B (raw)", 0, 255),
            (well_diff, "3. Normalised diff\n|A-B|/(A+B)", 0, 0.7),
            (well_smoothed, "4. Gaussian-smoothed", 0, well_smoothed.max()),
            (well_binary, f"5. Thresholded\n(> {noise_threshold})", 0, 1),
        ]
        for ax, (img, title, vmin, vmax) in zip(axes, panels):
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        plt.suptitle(
            f"Activity Analysis — Well {well.upper()} "
            f"(frames {frame_a_idx} \u2192 {frame_b_idx})",
            fontsize=13, y=1.05,
        )
        plt.tight_layout()
        fig.savefig(os.path.join(well_dir, "0_overview.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[Demo] Saved 6 well-{well.upper()} images to: {well_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate presentation images for each step of the "
                    "activity-analysis image transformation pipeline. "
                    "Read-only — the source video is never modified.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to the source .avi video (read-only).")
    parser.add_argument("--output_dir", default=None,
                        help="Folder for the generated demo images. Defaults to "
                             "<video_stem>_demo under " + DEFAULT_OUTPUT_ROOT + " "
                             "(never written into the video's own folder).")
    parser.add_argument("--frame_a", type=int, default=0, help="Index of the first frame.")
    parser.add_argument("--frame_b", type=int, default=None,
                        help="Index of the second frame. Defaults to frame_a + fps.")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Used to pick frame_b (frame_a + fps) when --frame_b is not given.")
    parser.add_argument("--noise_threshold", type=float, default=0.6,
                        help="Binarisation threshold (matches activity_analysis.py default).")
    parser.add_argument("--gaussian_std", type=float, default=0.9,
                        help="Sigma of the spatial smoothing kernel (matches activity_analysis.py default).")
    parser.add_argument("--roi", default=None,
                        help="Path to roi_info.json (from roi_detection.py / run_pipeline.py). "
                             "Required with --well.")
    parser.add_argument("--well", default=None,
                        help="Plate well label, e.g. D6 (row letter + 1-indexed column). "
                             "When given with --roi, also saves images cropped to that well.")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"[Demo] Video not found: {args.video}")

    if args.well and not args.roi:
        raise SystemExit("[Demo] --well requires --roi.")
    if args.roi and not os.path.exists(args.roi):
        raise SystemExit(f"[Demo] ROI file not found: {args.roi}")

    video_info = probe_video(args.video)
    num_frames = video_info["frame_count"]

    frame_a_idx = args.frame_a
    frame_b_idx = args.frame_b
    if frame_b_idx is None:
        frame_b_idx = min(frame_a_idx + int(round(args.fps)), num_frames - 1)

    output_dir = args.output_dir
    if output_dir is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, stem + "_demo")

    generate(
        video_path=args.video,
        output_dir=output_dir,
        frame_a_idx=frame_a_idx,
        frame_b_idx=frame_b_idx,
        noise_threshold=args.noise_threshold,
        gaussian_std=args.gaussian_std,
        roi_path=args.roi,
        well=args.well,
    )


if __name__ == "__main__":
    main()
