"""
run_pipeline.py — Chain roi_detection.py → activity_analysis.py → visualize_activity.py.

Runs the full Multiwell Swim Analysis workflow for a single video with one
command. All generated files (ROI, results, figures) are written to a
dedicated output folder created alongside the video —
<video_folder>/<video_stem>_output/ — instead of cluttering the raw-data
folder, so you don't have to retype matching paths for every step.

Each stage is invoked as a subprocess (python roi_detection.py ..., etc.), so
all of their existing interactive prompts (corner clicks, "happy with
parameters?" confirmations) still work exactly as before — this script only
removes the need to re-type consistent file paths between steps.

Steps:
  1. roi_detection.py     — skipped if the ROI file already exists (reused),
                             unless --force_roi is given.
  2. activity_analysis.py — skipped if the results file already exists
                             (reused), unless --force_activity is given.
  3. visualize_activity.py — always run, unless --skip_visualize is given.

Usage:
    python run_pipeline.py --video "D:/.../video0004 09-54-18.avi"
    # → creates D:/.../video0004 09-54-18_output/ containing roi_info.json,
    #   results.json, and (with --save) all figures.

    # Force re-running ROI detection and activity analysis, save all figures
    python run_pipeline.py --video "D:/.../video0004 09-54-18.avi" \\
        --force_roi --force_activity --save

    # Reuse a specific ROI file, custom activity-analysis parameters
    python run_pipeline.py --video "D:/.../video0004 09-54-18.avi" \\
        --roi "D:/.../shared_roi_info.json" \\
        --noise_threshold 0.5 --frame_skip 5 --save
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_DETECTION_SCRIPT = os.path.join(SCRIPT_DIR, "roi_detection.py")
ACTIVITY_ANALYSIS_SCRIPT = os.path.join(SCRIPT_DIR, "activity_analysis.py")
VISUALIZE_ACTIVITY_SCRIPT = os.path.join(SCRIPT_DIR, "visualize_activity.py")


def _default_output_dir(video_path: str) -> str:
    """Return <video_folder>/<video_stem>_output — created alongside the video
    to hold all generated files (ROI, results, figures) instead of cluttering
    the raw-data folder.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(video_path)), stem + "_output")


def _default_output_path(output_dir: str, video_path: str, suffix: str) -> str:
    """Return <output_dir>/<video_stem><suffix>."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, stem + suffix)


def _run_step(python_exe: str, script: str, args: list, step_name: str) -> None:
    """Run one pipeline stage as a subprocess, streaming its stdio directly."""
    cmd = [python_exe, script] + args
    print(f"\n[Pipeline] ── {step_name} ──")
    print("[Pipeline] " + " ".join(f'"{a}"' if " " in a else a for a in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"[Pipeline] '{step_name}' failed (exit code {result.returncode}). "
            "Stopping pipeline."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run roi_detection.py -> activity_analysis.py -> "
                    "visualize_activity.py in one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True,
                        help="Path to the video file (.avi).")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable to use for each stage.")
    parser.add_argument("--output_dir", default=None,
                        help="Folder to hold all generated files (ROI, results, "
                             "figures). Defaults to <video_stem>_output, created "
                             "alongside the video. Overridden per-file by --roi/--results.")

    # ── ROI detection ─────────────────────────────────────────────────────────
    parser.add_argument("--roi", default=None,
                        help="Path to roi_info.json. Defaults to "
                             "<output_dir>/<video_stem>_roi_info.json. "
                             "If this file already exists, ROI detection is "
                             "skipped and the file is reused.")
    parser.add_argument("--num_row", type=int, default=8, help="Number of plate rows.")
    parser.add_argument("--num_col", type=int, default=12, help="Number of plate columns.")
    parser.add_argument("--force_roi", action="store_true",
                        help="Re-run ROI detection even if the ROI file already exists.")

    # ── Activity analysis ─────────────────────────────────────────────────────
    parser.add_argument("--results", default=None,
                        help="Path to the results JSON. Defaults to "
                             "<output_dir>/<video_stem>_results.json. "
                             "If this file already exists, activity analysis is "
                             "skipped and the file is reused.")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Video frame rate (used for parameter preview only).")
    parser.add_argument("--noise_threshold", type=float, default=0.6,
                        help="Binarisation threshold for normalised diff.")
    parser.add_argument("--gaussian_std", type=float, default=0.9,
                        help="Sigma of the spatial smoothing kernel.")
    parser.add_argument("--frame_skip", type=int, default=5,
                        help="Frame offset for ActValS.")
    parser.add_argument("--force_activity", action="store_true",
                        help="Re-run activity analysis even if the results file already exists.")

    # ── Visualization ─────────────────────────────────────────────────────────
    parser.add_argument("--skip_visualize", action="store_true",
                        help="Skip the visualize_activity.py step.")
    parser.add_argument("--save", action="store_true",
                        help="Pass --save through to visualize_activity.py "
                             "(saves PNGs alongside the results file).")

    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        raise SystemExit(f"[Pipeline] Video not found: {video_path}")

    output_dir = args.output_dir or _default_output_dir(video_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Pipeline] Output folder: {output_dir}")

    # ── Step 1: ROI detection ─────────────────────────────────────────────────
    roi_path = args.roi or _default_output_path(output_dir, video_path, "_roi_info.json")
    if args.force_roi or not os.path.exists(roi_path):
        _run_step(
            args.python, ROI_DETECTION_SCRIPT,
            ["--video", video_path, "--output", roi_path,
             "--num_row", str(args.num_row), "--num_col", str(args.num_col)],
            "Step 1: ROI Detection",
        )
    else:
        print(f"\n[Pipeline] Reusing existing ROI file: {roi_path}")
        print("[Pipeline] (pass --force_roi to re-run ROI detection)")

    # ── Step 2: Activity analysis ─────────────────────────────────────────────
    results_path = args.results or _default_output_path(output_dir, video_path, "_results.json")
    if args.force_activity or not os.path.exists(results_path):
        _run_step(
            args.python, ACTIVITY_ANALYSIS_SCRIPT,
            ["--video", video_path, "--roi", roi_path, "--output", results_path,
             "--fps", str(args.fps),
             "--noise_threshold", str(args.noise_threshold),
             "--gaussian_std", str(args.gaussian_std),
             "--frame_skip", str(args.frame_skip)],
            "Step 2: Activity Analysis",
        )
    else:
        print(f"\n[Pipeline] Reusing existing results file: {results_path}")
        print("[Pipeline] (pass --force_activity to re-run activity analysis)")

    # ── Step 3: Visualization ─────────────────────────────────────────────────
    if not args.skip_visualize:
        viz_args = ["--results", results_path, "--video", video_path, "--roi", roi_path]
        if args.save:
            viz_args.append("--save")
        _run_step(args.python, VISUALIZE_ACTIVITY_SCRIPT, viz_args, "Step 3: Visualization")
    else:
        print("\n[Pipeline] Skipping visualization (--skip_visualize).")

    print("\n[Pipeline] Done.")
    print(f"  ROI file     : {roi_path}")
    print(f"  Results file : {results_path}")


if __name__ == "__main__":
    main()
