"""
roi_detection.py — Step 1: ROI Detection for Multiwell Swim Analysis.

Replicates MATLAB HJ_MultWell_Analysis.m, Step 1 (Auto Selection method).

MATLAB original logic:
  1. Show first frame of a sample video.
  2. User draws circles on the 4 corner wells (UL, UR, LL, LR).
  3. Extract center (x, y) and radius for each corner.
  4. Radius = mean of the 4 corner radii.
  5. Interpolate an 8×12 grid of well centres via bilinear mapping:
       Lx[r] = ULx + (LLx-ULx)/(NumRow-1) * r     (left-edge x per row)
       Ly[r] = ULy + (LLy-ULy)/(NumRow-1) * r
       Rx[r] = URx + (LRx-URx)/(NumRow-1) * r     (right-edge x per row)
       Ry[r] = URy + (LRy-URy)/(NumRow-1) * r
       center[r,c] = ( Lx[r] + (Rx[r]-Lx[r])/(NumCol-1)*c ,
                       Ly[r] + (Ry[r]-Ly[r])/(NumCol-1)*c )
  6. Display overlay and ask user to confirm.
  7. Save to ROI_Info.mat  →  here saved as roi_info.json.

Usage:
  New ROI:
    python roi_detection.py --video PATH [--output roi_info.json]
                            [--num_row 8] [--num_col 12]

  Load & display existing ROI:
    python roi_detection.py --video PATH --load existing_roi.json
                            [--output roi_info.json]
"""

import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from utils import load_roi, save_roi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_first_frame(video_path: str) -> np.ndarray:
    """Return the first frame as float32 grayscale (simple channel average)."""
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


def _interpolate_grid(ul, ur, ll, lr, radius: float,
                      num_row: int, num_col: int) -> list:
    """Interpolate ROI centres from 4 corner centres.

    Mirrors MATLAB exactly:
        Lx = ULx + (LLx-ULx)/(NumRow-1)*(0:NumRow-1)
        Ly = ULy + (LLy-ULy)/(NumRow-1)*(0:NumRow-1)
        Rx = URx + (LRx-URx)/(NumRow-1)*(0:NumRow-1)
        Ry = URy + (LRy-URy)/(NumRow-1)*(0:NumRow-1)
        Center[r,c] = [ Lx[r]+(Rx[r]-Lx[r])/(NumCol-1)*c ,
                        Ly[r]+(Ry[r]-Ly[r])/(NumCol-1)*c ]

    All corner inputs are (x, y) tuples (matplotlib ginput convention).
    Radius is a single float shared by all wells (MATLAB: mean of 4 corner radii).

    Returns:
        list of dicts ordered row-major: (r=0,c=0) … (r=NumRow-1,c=NumCol-1),
        each dict has 'center': [x, y] and 'radius': float.
    """
    ULx, ULy = ul
    URx, URy = ur
    LLx, LLy = ll
    LRx, LRy = lr

    rows = np.arange(num_row, dtype=np.float64)

    # Left-edge column centres for every row
    Lx = ULx + (LLx - ULx) / (num_row - 1) * rows
    Ly = ULy + (LLy - ULy) / (num_row - 1) * rows

    # Right-edge column centres for every row
    Rx = URx + (LRx - URx) / (num_row - 1) * rows
    Ry = URy + (LRy - URy) / (num_row - 1) * rows

    roi_list = []
    for r in range(num_row):
        for c in range(num_col):
            cx = Lx[r] + (Rx[r] - Lx[r]) / (num_col - 1) * c
            cy = Ly[r] + (Ry[r] - Ly[r]) / (num_col - 1) * c
            roi_list.append({
                "center": [float(cx), float(cy)],
                "radius": float(radius),
            })

    return roi_list


def _draw_overlay(ax, roi_list: list, corner_pts=None) -> None:
    """Overlay ROI circles and markers on a matplotlib Axes.

    Mirrors MATLAB:
        plot(Center(1), Center(2), '*r')
        drawcircle('Center', Center, 'Radius', Radius)   → red circle
        plot(ULx,ULy,'*m') ... (magenta star for each corner)
    """
    for roi in roi_list:
        cx, cy = roi["center"]
        r = roi["radius"]
        circle = plt.Circle((cx, cy), r, color="red", fill=False, linewidth=0.5)
        ax.add_patch(circle)
        ax.plot(cx, cy, "*r", markersize=2, markeredgewidth=0.5)

    if corner_pts:
        for (x, y) in corner_pts:
            ax.plot(x, y, "*m", markersize=10, markeredgewidth=1.0)


# ---------------------------------------------------------------------------
# Main modes
# ---------------------------------------------------------------------------

def run_new_roi(frame: np.ndarray, num_row: int, num_col: int,
                output_path: str) -> list:
    """Interactive new-ROI workflow: 4 corner clicks → radius input → confirm.

    Mirrors MATLAB Auto Selection (case 1), but instead of drawing circles
    the user clicks the centre of each corner well and types the radius.
    """
    roi_list = None
    happy = False

    while not happy:
        # ── Step A: collect 4 corner clicks ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.imshow(frame, cmap="gray", aspect="equal", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            "Click the centre of each corner well in order:\n"
            "  1) UPPER-LEFT   2) UPPER-RIGHT   3) LOWER-LEFT   4) LOWER-RIGHT\n"
            "Press ENTER after 4 clicks.",
            fontsize=11,
        )
        plt.tight_layout()
        fig.canvas.manager.set_window_title("ROI Detection — Corner Selection")

        print("\n[ROI] Waiting for 4 corner-well clicks (UL → UR → LL → LR)…")
        pts = plt.ginput(4, timeout=-1)   # blocks until 4 clicks + Enter
        plt.close(fig)

        if len(pts) < 4:
            print("[ROI] Need exactly 4 clicks. Please try again.")
            continue

        ul, ur, ll, lr = pts[0], pts[1], pts[2], pts[3]
        print(f"  UL : ({ul[0]:.1f},  {ul[1]:.1f})")
        print(f"  UR : ({ur[0]:.1f},  {ur[1]:.1f})")
        print(f"  LL : ({ll[0]:.1f},  {ll[1]:.1f})")
        print(f"  LR : ({lr[0]:.1f},  {lr[1]:.1f})")

        # ── Step B: get radius ───────────────────────────────────────────────
        # MATLAB: Radius = mean([ULr URr LLr LRr])  (mean of 4 drawn-circle radii)
        # Here: single typed value (same radius for all wells, per design choice).
        while True:
            try:
                raw = input("\n[ROI] Enter well radius in pixels [default 30]: ").strip()
                radius = float(raw) if raw else 30.0
                if radius <= 0:
                    print("  Radius must be positive.")
                    continue
                break
            except ValueError:
                print("  Please enter a number.")

        # ── Step C: interpolate grid ─────────────────────────────────────────
        roi_list = _interpolate_grid(ul, ur, ll, lr, radius, num_row, num_col)

        # ── Step D: preview overlay ──────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(13, 8))
        ax2.imshow(frame, cmap="gray", aspect="equal", interpolation="nearest")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title(
            f"ROI Preview — {num_row}×{num_col} = {num_row * num_col} wells  "
            f"|  Radius = {radius:.1f} px  |  Close window to continue",
            fontsize=11,
        )
        _draw_overlay(ax2, roi_list, corner_pts=[ul, ur, ll, lr])
        plt.tight_layout()
        fig2.canvas.manager.set_window_title("ROI Detection — Preview")
        plt.show()

        # ── Step E: confirm ──────────────────────────────────────────────────
        ans = input("\n[ROI] Happy with the ROI? (1 = yes, 0 = redo): ").strip()
        happy = ans == "1"
        if not happy:
            print("[ROI] Restarting ROI selection…\n")

    save_roi(roi_list, num_row, num_col, output_path)
    print(f"[ROI] Saved to: {output_path}")
    return roi_list


def run_load_roi(frame: np.ndarray, load_path: str, output_path: str) -> list:
    """Load an existing roi_info.json, display overlay, optionally re-save.

    Mirrors MATLAB case 2 (Load ROI File).
    """
    data = load_roi(load_path)
    roi_list = data["roi"]
    num_row = data["num_row"]
    num_col = data["num_col"]

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(frame, cmap="gray", aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Loaded ROI — {num_row}×{num_col} = {len(roi_list)} wells  "
        "|  Close window to continue",
        fontsize=11,
    )
    _draw_overlay(ax, roi_list)
    plt.tight_layout()
    fig.canvas.manager.set_window_title("ROI Detection — Loaded ROI")
    plt.show()

    # Optionally re-save to a different path
    if output_path and output_path != load_path:
        save_roi(roi_list, num_row, num_col, output_path)
        print(f"[ROI] Re-saved to: {output_path}")

    print(f"[ROI] Loaded successfully ({len(roi_list)} wells).")
    return roi_list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ROI Detection for Multiwell Swim Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True,
                        help="Path to a sample video file (.avi).")
    parser.add_argument("--output", default="roi_info.json",
                        help="Output path for roi_info.json.")
    parser.add_argument("--load", default=None,
                        help="Load an existing roi_info.json instead of picking a new ROI.")
    parser.add_argument("--num_row", type=int, default=8,
                        help="Number of plate rows.")
    parser.add_argument("--num_col", type=int, default=12,
                        help="Number of plate columns.")
    args = parser.parse_args()

    print(f"[ROI] Reading first frame from: {args.video}")
    frame = _read_first_frame(args.video)
    print(f"[ROI] Image size: {frame.shape[1]} x {frame.shape[0]} (W x H)")

    if args.load:
        run_load_roi(frame, args.load, args.output)
    else:
        run_new_roi(frame, args.num_row, args.num_col, args.output)

    print("[ROI] Done.")


if __name__ == "__main__":
    main()
