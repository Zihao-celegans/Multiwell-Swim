"""
activity_analysis.py — Step 2: Activity Analysis for Multiwell Swim Analysis.

Replicates MATLAB HJ_MultWell_Analysis.m, "Main Activity Analysis" section.

───────────────────────────────────────────────────────────────────────────────
MATLAB original logic (faithfully mirrored below):

1. Build Gaussian spatial-filter kernel:
       x = -5:5;  y = x;
       [xx, yy] = meshgrid(x, y);
       gau = exp(-sqrt(xx.^2 + yy.^2) / GaussianStd^2)
   Note: exponent is  -r / σ²  (NOT the standard -r²/(2σ²)).
   Default GaussianStd = 0.9.  Kernel is 11×11, not normalised.

2. Parameter-preview loop (mirrors MATLAB "are you happy" loop):
       sample1 = frame 1 (index 0),  sample2 = frame 1+fps (index fps)
       Diff1 = |sample1 - sample2| / |sample1 + sample2|
       smoothedDiff1 = convn(Diff1, gau, 'same')
       BinaryDiff1   = smoothedDiff1 > NoiseThres
   Show raw diff (clamped to [0, 0.7]) and thresholded image side-by-side.
   Repeat until user is satisfied.

3. Main frame loop (NumA = 1 … NumFrames, MATLAB 1-indexed):
       ImgA  = current frame
       ImgB  = frame at NumA-1        (previous frame)
       ImgC  = frame at NumA-frameSkip  (only when NumA > 1+frameSkip)

       ImgDif     = abs(ImgA - ImgB) / (ImgA + ImgB)
       ImgDifSkip = abs(ImgA - ImgC) / (ImgA + ImgC)

   For each ROI n:
       Center = round(ROI(n).Center)          → integer (xc, yc)
       R      = round(ROI(n).Radius)          → integer
       ROIDif = ImgDif[yc-R:yc+R, xc-R:xc+R] → square crop, (2R+1)×(2R+1)
       ROIDifT = conv2(ROIDif, gau, 'same') > NoiseThres
       ActVal(n, NumA-1)  = sum(ROIDifT)      (stored at transition index)

       Same with ImgDifSkip → ActValS(n, NumA-1)   (when NumA > 1+frameSkip)

       Deviation from raw MATLAB: if the (2R+1)×(2R+1) crop extends past
       the frame edges (e.g. plate row placed a few pixels beyond the
       image boundary), the out-of-frame portion is zero-filled ("no
       activity") instead of discarding the whole well with NaN.

4. Average over frame-transitions:
       ActVal_avg  = mean(ActVal,  2, 'omitnan')   → (NumROI,)
       ActValS_avg = mean(ActValS, 2, 'omitnan')   → (NumROI,)
       Reshape to (NumRow, NumCol):
         MATLAB: reshape([NumCol, NumRow])' (column-major + transpose)
         Python: .reshape(num_row, num_col)  (row-major; equivalent result)

5. Output mirrors MATLAB struct S fields (one entry per video):
       Vidname, FrameRate, FrameSkip, ActVal (8×12), ActValS (8×12), date
       Future fields (Freq, EggArea, …) stored as null.

───────────────────────────────────────────────────────────────────────────────

Usage:
    python activity_analysis.py \\
        --video  "D:/MultiWell_swim/.../video0010 12-04-18.avi" \\
        --roi    roi_info.json \\
        [--output results.json] \\
        [--fps 5] [--noise_threshold 0.6] [--gaussian_std 0.9] [--frame_skip 5]
"""

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

from utils import load_roi, probe_video, read_video_frame, iter_video_frames_grayscale


# ---------------------------------------------------------------------------
# Gaussian kernel
# ---------------------------------------------------------------------------

def build_gaussian_kernel(gaussian_std: float) -> np.ndarray:
    """Build the 11×11 spatial-smoothing kernel.

    Mirrors MATLAB exactly:
        x = -5:5;  y = x;
        [xx, yy] = meshgrid(x, y);
        gau = exp(-sqrt(xx.^2 + yy.^2) / GaussianStd^2)

    The exponent is  -r / σ²  where  r = sqrt(x²+y²).
    This is NOT a standard Gaussian (which would be -r²/(2σ²)).
    The kernel is also NOT normalised (values do not sum to 1).
    """
    x = np.arange(-5, 6, dtype=np.float64)   # [-5, -4, …, 4, 5]  (11 elements)
    xx, yy = np.meshgrid(x, x)               # same convention as MATLAB meshgrid
    gau = np.exp(-np.sqrt(xx ** 2 + yy ** 2) / gaussian_std ** 2)
    return gau


# ---------------------------------------------------------------------------
# Normalised frame difference
# ---------------------------------------------------------------------------

def _normalised_diff(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Compute  |A − B| / (A + B)  pixel-wise (float64 inputs expected).

    Mirrors MATLAB:
        ImgDif = abs(double(ImgA - ImgB)) ./ double(ImgA + ImgB)

    Where A+B == 0 the result is NaN (matching MATLAB behaviour).
    Downstream thresholding treats NaN as 0 (NaN > threshold → False),
    so these pixels do not contribute to the activity count.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.abs(img_a - img_b) / (img_a + img_b)
    return diff


# ---------------------------------------------------------------------------
# Per-ROI activity computation
# ---------------------------------------------------------------------------

def _roi_activity(img_dif: np.ndarray, roi: dict,
                  gau: np.ndarray, noise_threshold: float) -> float:
    """Compute activity count for one ROI from a normalised difference image.

    Mirrors MATLAB:
        Center = round(ROI(n).Center);   xc = Center(1);  yc = Center(2);
        R      = round(ROI(n).Radius);
        ROIDif  = ImgDif(yc-R:yc+R, xc-R:xc+R);        % square crop (2R+1)×(2R+1)
        ROIDifT = conv2(ROIDif, gau, 'same');
        ROIDifT = ROIDifT > NoiseThres;
        ActVal(n) = sum(ROIDifT(:), 'omitnan');          % count active pixels

    conv2(A, B, 'same') uses zero-padding at borders →
        scipy.signal.convolve2d(A, B, mode='same', boundary='fill', fillvalue=0).

    NaN values in img_dif propagate through the convolution, then
    NaN > noise_threshold evaluates to False (0), matching MATLAB behaviour.

    Wells whose (2R+1)×(2R+1) crop extends past the image edges (e.g. a
    plate row placed a few pixels beyond the frame boundary) are handled by
    treating the out-of-frame portion of the crop as zero — i.e. "no
    activity" — rather than discarding the whole well. This is an
    intentional deviation from raw MATLAB (which would raise an index
    error); only the truly out-of-frame pixels are zero-filled, the
    in-frame pixels are used as-is.

    Returns NaN only if the ROI's centre + radius crop does not overlap the
    image at all (i.e. the well is entirely outside the frame).
    """
    xc = int(round(roi["center"][0]))
    yc = int(round(roi["center"][1]))
    R  = int(round(roi["radius"]))

    img_h, img_w = img_dif.shape
    size = 2 * R + 1

    # Desired crop window — may extend past the image edges.
    y0, y1 = yc - R, yc + R + 1          # exclusive end
    x0, x1 = xc - R, xc + R + 1

    # Clip to the valid image range.
    y0_clip, y1_clip = max(y0, 0), min(y1, img_h)
    x0_clip, x1_clip = max(x0, 0), min(x1, img_w)

    if y0_clip >= y1_clip or x0_clip >= x1_clip:
        # Well is entirely outside the frame — nothing to measure.
        return np.nan

    # Zero-fill the full (2R+1)x(2R+1) window, then paste in whatever part
    # of the crop actually falls inside the image. Pixels outside the frame
    # stay 0 (treated as "no activity").
    roi_dif = np.zeros((size, size), dtype=img_dif.dtype)
    py0, px0 = y0_clip - y0, x0_clip - x0
    py1, px1 = py0 + (y1_clip - y0_clip), px0 + (x1_clip - x0_clip)
    roi_dif[py0:py1, px0:px1] = img_dif[y0_clip:y1_clip, x0_clip:x1_clip]

    # 2-D convolution, zero-padded — mirrors MATLAB conv2(..., 'same')
    roi_smoothed = convolve2d(roi_dif, gau,
                              mode="same", boundary="fill", fillvalue=0.0)

    # Threshold: NaN > noise_threshold → False (0) in both MATLAB and NumPy
    roi_binary = roi_smoothed > noise_threshold      # bool array

    # sum without omitnan: boolean array has no NaN; mirrors MATLAB sum(logical)
    return float(np.sum(roi_binary))


# ---------------------------------------------------------------------------
# Parameter preview (mirrors MATLAB "are you happy" loop)
# ---------------------------------------------------------------------------

def _parameter_preview(video_path: str, num_frames: int, fps: int,
                        gau: np.ndarray, noise_threshold: float) -> None:
    """Show raw normalised diff and thresholded diff for a representative frame pair.

    Mirrors MATLAB parameter-check block:
        sample1 = double(im2gray(read(sampleVid, 1)))          → frame 0
        sample2 = double(im2gray(read(sampleVid, 1 + fps)))    → frame fps
        Diff1 = abs(sample1 - sample2) ./ abs(sample1 + sample2)
        smoothedDiff1 = convn(Diff1, gau, 'same')
        BinaryDiff1   = smoothedDiff1 > NoiseThres
        imagesc(Diff1, [0 0.7])   then   imagesc(BinaryDiff1)

    convn on a 2-D array is equivalent to conv2, both zero-padded 'same'.
    For the full-image preview we use nan_to_num(0) before convolution
    because the image is large and sparse NaN propagation would hide activity;
    this preview is only for visual parameter tuning and does not affect output.

    Only the two needed frames are read (via seek), not the whole video.
    """
    frame_b_idx = min(fps, num_frames - 1)

    sample1 = read_video_frame(video_path, 0).astype(np.float64)
    sample2 = read_video_frame(video_path, frame_b_idx).astype(np.float64)

    diff1 = _normalised_diff(sample1, sample2)

    # Replace NaN with 0 for the full-image convolution (preview only)
    diff1_clean = np.nan_to_num(diff1, nan=0.0)
    smoothed = convolve2d(diff1_clean, gau,
                          mode="same", boundary="fill", fillvalue=0.0)
    binary = smoothed > noise_threshold

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(diff1, cmap="gray", vmin=0, vmax=0.7,
                   aspect="equal", interpolation="nearest")
    axes[0].set_title(
        f"Normalised diff  |A−B|/(A+B)   [display range 0–0.7]\n"
        f"frames 0 and {frame_b_idx}",
        fontsize=10,
    )
    axes[0].axis("off")

    axes[1].imshow(binary, cmap="gray", aspect="equal", interpolation="nearest")
    axes[1].set_title(
        f"Thresholded   (noise_threshold = {noise_threshold:.3f})",
        fontsize=10,
    )
    axes[1].axis("off")

    plt.suptitle("Parameter Preview  |  Close window to continue", fontsize=12)
    plt.tight_layout()
    fig.canvas.manager.set_window_title("Activity Analysis — Parameter Preview")
    plt.show()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(video_path: str, roi_path: str, output_path: str,
            fps: float, noise_threshold: float,
            gaussian_std: float, frame_skip: int) -> dict:
    """Run the full activity analysis pipeline and return the result dict.

    Parameters
    ----------
    video_path      : path to the .avi video file.
    roi_path        : path to roi_info.json.
    output_path     : path for the JSON output.
    fps             : frame rate (used for the parameter preview only;
                      does not affect frame-by-frame analysis).
    noise_threshold : binarisation threshold (MATLAB default 0.6).
    gaussian_std    : kernel parameter (MATLAB default 0.9).
    frame_skip      : offset for ActValS (MATLAB default 5).
    """

    # ── Load ROI ─────────────────────────────────────────────────────────────
    print(f"[Activity] Loading ROI from: {roi_path}")
    roi_data  = load_roi(roi_path)
    roi_list  = roi_data["roi"]
    num_row   = roi_data["num_row"]
    num_col   = roi_data["num_col"]
    num_roi   = len(roi_list)
    print(f"[Activity]   {num_row}×{num_col} = {num_roi} wells")

    # ── Probe video (metadata only — does not decode/load frames) ────────────
    print(f"[Activity] Reading video info: {video_path}")
    video_info = probe_video(video_path)
    num_frames_estimate = video_info["frame_count"]
    print(f"[Activity]   ~{num_frames_estimate} frames (estimate),  "
          f"image size {video_info['width']}×{video_info['height']} (W×H)")

    # ── Parameter-preview loop ────────────────────────────────────────────────
    # Mirrors MATLAB:
    #   Build gau, show preview, ask "are you happy?", loop if not.
    fps_int = int(round(fps))
    happy = False
    while not happy:
        gau = build_gaussian_kernel(gaussian_std)
        _parameter_preview(video_path, num_frames_estimate, fps_int, gau, noise_threshold)

        print("\n[Activity] Current parameters:")
        print(f"  fps             = {fps}")
        print(f"  noise_threshold = {noise_threshold}")
        print(f"  gaussian_std    = {gaussian_std}")
        print(f"  frame_skip      = {frame_skip}")

        ans = input(
            "\n[Activity] Happy with parameters? (1 = yes, 0 = change): "
        ).strip()
        if ans == "1":
            happy = True
        else:
            # Re-prompt for adjustable parameters
            # (fps and frame_skip cannot be changed here without reloading frames)
            try:
                val = input(
                    f"  noise_threshold  [{noise_threshold}]  (press Enter to keep): "
                ).strip()
                if val:
                    noise_threshold = float(val)

                val = input(
                    f"  gaussian_std     [{gaussian_std}]  (press Enter to keep): "
                ).strip()
                if val:
                    gaussian_std = float(val)
                    gau = build_gaussian_kernel(gaussian_std)
            except ValueError:
                print("  Invalid input — keeping current values.")

    # ── Main frame loop ───────────────────────────────────────────────────────
    # MATLAB:  NumA = 1 … NumFrames  (1-indexed)
    # Python:  frame_idx = 0 … num_frames-1  (0-indexed),  NumA = frame_idx + 1
    #
    # Frames are streamed one at a time (iter_video_frames_grayscale) instead
    # of loading the whole video into RAM. ImgC needs the frame `frame_skip`
    # steps back, so a rolling buffer of the last `frame_skip` frames (float32)
    # is kept instead of the full array — memory use no longer scales with
    # video length/resolution.
    #
    # Columns are collected as they're computed (num_frames isn't known exactly
    # ahead of time — CAP_PROP_FRAME_COUNT is only an estimate for some
    # codecs/containers) and stacked into arrays once streaming finishes.
    print("\n[Activity] Starting frame-by-frame analysis…")

    act_val_cols = []      # each: (num_roi,) — one per frame transition
    act_val_s_cols = []    # aligned index-for-index with act_val_cols
    buffer = deque(maxlen=max(frame_skip, 1))   # previous `frame_skip` frames, oldest first

    frame_idx = 0
    for frame in iter_video_frames_grayscale(video_path):
        img_a = frame.astype(np.float64)

        if frame_idx >= 1:
            # Previous frame (ImgB) — float64 for diff
            img_b = buffer[-1].astype(np.float64)

            # Normalised difference for adjacent frames
            # MATLAB: ImgDif = abs(double(ImgA - ImgB)) ./ double(ImgA + ImgB)
            img_dif = _normalised_diff(img_a, img_b)

            act_val_cols.append(np.array(
                [_roi_activity(img_dif, roi, gau, noise_threshold) for roi in roi_list]
            ))

            # ActValS: compare current frame with frame `frame_skip` steps earlier
            # MATLAB condition: NumA > 1 + frameSkip  →  frame_idx > frame_skip
            if frame_idx > frame_skip:
                img_c = buffer[0].astype(np.float64)

                # MATLAB: ImgDifSkip = abs(double(ImgA - ImgC)) ./ double(ImgA + ImgC)
                img_dif_skip = _normalised_diff(img_a, img_c)

                act_val_s_cols.append(np.array(
                    [_roi_activity(img_dif_skip, roi, gau, noise_threshold) for roi in roi_list]
                ))
            else:
                act_val_s_cols.append(np.full(num_roi, np.nan))

            # Progress (mirrors MATLAB fprintf per frame)
            if frame_idx % 10 == 0:
                print(f"  [Activity] Frame transition {frame_idx:4d}", end="\r")

        buffer.append(frame)   # kept as float32 to minimise buffer memory
        frame_idx += 1

    num_frames = frame_idx   # actual count, now that streaming is done
    print(f"\n[Activity] Complete — processed {num_frames - 1} frame transitions.")

    # ── Stack into (num_roi, num_frames) arrays ───────────────────────────────
    # Mirrors MATLAB: ActVal = nan(NumROI, NumFrames)
    # Column indices 0 … num_frames-2 are filled (one per frame transition).
    # Column num_frames-1 stays NaN and is excluded by nanmean.
    if act_val_cols:
        act_val   = np.column_stack(act_val_cols)
        act_val_s = np.column_stack(act_val_s_cols)
    else:
        act_val   = np.empty((num_roi, 0))
        act_val_s = np.empty((num_roi, 0))
    act_val   = np.hstack([act_val,   np.full((num_roi, 1), np.nan)])
    act_val_s = np.hstack([act_val_s, np.full((num_roi, 1), np.nan)])

    # ── Average over frame transitions, reshape to (NumRow, NumCol) ───────────
    # MATLAB:
    #   ActVal_avg  = mean(ActVal, 2, 'omitnan')           → (NumROI, 1) col-vector
    #   ActVal_avg_reshape = reshape(ActVal_avg, [NumCol, NumRow])'  → (NumRow, NumCol)
    #
    # Python equivalent:
    #   np.nanmean(act_val, axis=1)        → (num_roi,)
    #   .reshape(num_row, num_col)         → (num_row, num_col)  [row-major = same result]
    #
    # Proof of equivalence:
    #   MATLAB fills reshape([NumCol,NumRow]) column-major → element (i,j) = v[(j-1)*NumCol+i-1]
    #   Transpose → element [r,c] = v[(r-1)*NumCol + c-1]
    #   Since ROIs are stored row-major n = (r-1)*NumCol + c, this equals v[n]. ✓
    act_val_avg   = np.nanmean(act_val,   axis=1)      # (num_roi,)
    act_val_s_avg = np.nanmean(act_val_s, axis=1)      # (num_roi,)

    act_val_2d   = act_val_avg.reshape(num_row, num_col)     # (8, 12)
    act_val_s_2d = act_val_s_avg.reshape(num_row, num_col)   # (8, 12)

    # ── Build result dict (mirrors MATLAB struct S) ───────────────────────────
    vid_mtime = os.path.getmtime(video_path)
    vid_date  = datetime.fromtimestamp(vid_mtime).strftime("%d-%b-%Y %H:%M:%S")

    # ── Full per-frame timecourse (num_roi × num_frames) ────────────────────
    # Stored as a list-of-lists (one list per well, length = num_frames).
    # NaN encoded as null for JSON compatibility.
    # Only the valid transition columns (0 … num_frames-2) carry real data;
    # the last column is NaN and is included so the shape is unambiguous.
    def _to_json_list(arr2d):
        """Convert a 2-D float array to a nested list, NaN → None."""
        return [
            [None if np.isnan(v) else float(v) for v in row]
            for row in arr2d
        ]

    result = {
        # Core fields (mirrors MATLAB struct S)
        "Vidname"   : os.path.basename(video_path),
        "FrameRate" : fps,
        "FrameSkip" : frame_skip,   # NOTE: MATLAB has a bug where FrameRate is
                                     # overwritten by frameSkip; we store both correctly.
        "ActVal"    : act_val_2d.tolist(),    # 8×12 list-of-lists (mean per well)
        "ActValS"   : act_val_s_2d.tolist(),  # 8×12 list-of-lists (mean per well)
        # Full timecourse: num_roi lists, each of length num_frames.
        # Index order matches roi_list (row-major: A1, A2, …, H12).
        # Time axis: frame index t → time (s) = t / fps.
        "ActValS_timecourse": _to_json_list(act_val_s),  # (num_roi, num_frames)
        "date"      : vid_date,
        # Reserved for future steps (Step 3 & 4) — stored as null
        "Freq"      : None,
        "Freq_L"    : None,
        "Freq_U"    : None,
        "Fullseq"   : None,
        "Periodicity": None,
        "EggArea"   : None,
        "EggMaskNum": None,
        # Extra metadata
        "noise_threshold": noise_threshold,
        "gaussian_std"   : gaussian_std,
    }

    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[Activity] Results saved to: {output_path}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Activity Analysis for Multiwell Swim Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True,
                        help="Path to the video file (.avi).")
    parser.add_argument("--roi", required=True,
                        help="Path to roi_info.json.")
    parser.add_argument("--output", default=None,
                        help="Output path for results.json. "
                             "Defaults to <video_stem>_results.json alongside the video.")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Video frame rate (used for parameter preview frame pair).")
    parser.add_argument("--noise_threshold", type=float, default=0.6,
                        help="Binarisation threshold for normalised diff (MATLAB default 0.6).")
    parser.add_argument("--gaussian_std", type=float, default=0.9,
                        help="σ of spatial smoothing kernel (MATLAB default 0.9).")
    parser.add_argument("--frame_skip", type=int, default=5,
                        help="Frame offset for ActValS (MATLAB default 5).")
    args = parser.parse_args()

    # Default output path: same folder as the video
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(args.video)),
            stem + "_results.json",
        )

    analyse(
        video_path      = args.video,
        roi_path        = args.roi,
        output_path     = args.output,
        fps             = args.fps,
        noise_threshold = args.noise_threshold,
        gaussian_std    = args.gaussian_std,
        frame_skip      = args.frame_skip,
    )


if __name__ == "__main__":
    main()
