"""
utils.py — Shared utilities for Multiwell Swim Analysis.

Provides:
  - load_roi  / save_roi           : JSON-based ROI file I/O
  - read_video_grayscale           : load all frames as float32 grayscale
  - probe_video                    : read video metadata without decoding frames
  - read_video_frame                : seek + read a single frame
  - iter_video_frames_grayscale    : stream frames one at a time (low memory)

Grayscale conversion mirrors MATLAB HJ_MultWell_Analysis.m:
  RGB video   → single(squeeze(mean(AllFrames, 3)))   (simple channel average)
  Grayscale   → single(squeeze(AllFrames))            (direct, no conversion)
"""

import json
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# ROI file I/O
# ---------------------------------------------------------------------------

def load_roi(json_path: str) -> dict:
    """Load ROI info from a JSON file.

    Returns a dict with keys:
        'num_row'  : int   — number of plate rows  (typically 8)
        'num_col'  : int   — number of plate columns (typically 12)
        'roi'      : list of dicts, each with:
                       'center'  : [x, y]  (float, image pixel coordinates)
                       'radius'  : float   (pixels)
                     Ordered row-major: (r=0,c=0), (r=0,c=1), ..., (r=0,c=NumCol-1),
                     (r=1,c=0), ..., (r=NumRow-1,c=NumCol-1)
    """
    with open(json_path, "r") as fh:
        return json.load(fh)


def save_roi(roi_list: list, num_row: int, num_col: int, json_path: str) -> None:
    """Save ROI info to a JSON file.

    Args:
        roi_list  : list of dicts with 'center': [x, y] and 'radius': float,
                    ordered row-major (see load_roi).
        num_row   : number of plate rows.
        num_col   : number of plate columns.
        json_path : output file path.
    """
    data = {
        "num_row": num_row,
        "num_col": num_col,
        "roi": roi_list,
    }
    with open(json_path, "w") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Video reading
# ---------------------------------------------------------------------------

def _to_gray_float32(frame: np.ndarray) -> np.ndarray:
    """Convert one cv2-decoded frame (BGR or gray, uint8) to float32 grayscale.

    MATLAB mean(AllFrames, 3) = simple average over colour channels; result is
    identical regardless of channel ordering (BGR vs RGB).
    """
    if frame.ndim == 3:
        return frame[:, :, :3].mean(axis=2).astype(np.float32)
    return frame.astype(np.float32)


def read_video_grayscale(video_path: str) -> np.ndarray:
    """Read all frames from a video file and return as float32 grayscale.

    Mirrors MATLAB:
        RGB   : AllFrames = single(squeeze(mean(AllFrames, 3)))
                → simple arithmetic mean of colour channels (not weighted luminance)
        Gray  : AllFrames = single(squeeze(AllFrames))

    Loads the entire video into memory at once — for large/high-resolution
    videos prefer `iter_video_frames_grayscale`, which streams one frame at a
    time instead.

    Returns:
        np.ndarray of shape (H, W, N_frames), dtype=float32.

    Raises:
        IOError   if the file cannot be opened.
        ValueError if no frames are read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(_to_gray_float32(frame))

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")

    return np.stack(frames, axis=2)   # (H, W, N_frames)


def probe_video(video_path: str) -> dict:
    """Read video metadata (width, height, frame_count, fps) without decoding frames.

    `frame_count` comes from container metadata and can be inaccurate for some
    codecs/containers — treat it as an estimate (e.g. for progress display),
    not a guaranteed exact count.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()
    return info


def read_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Read a single frame (float32 grayscale) by seeking, without loading the rest."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from: {video_path}")
    return _to_gray_float32(frame)


def iter_video_frames_grayscale(video_path: str):
    """Yield frames one at a time as float32 grayscale (same conversion as
    `read_video_grayscale`), so memory use stays O(1) regardless of video
    length/resolution instead of holding the whole video in RAM at once.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield _to_gray_float32(frame)
    finally:
        cap.release()
