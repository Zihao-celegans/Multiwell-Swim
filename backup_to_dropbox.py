"""
backup_to_dropbox.py — Incrementally back up processed data (not raw videos)
from an external hard drive to a Dropbox folder for cloud backup.

Mirrors the folder structure of the source tree, but skips raw video files
(.avi by default) so only ROI files, results JSON, figures, and other
processed/analysis outputs get copied. Each physical drive gets its own
subfolder under --dest, so multiple hard drives reusing the same drive
letter (e.g. D:) don't collide or overwrite each other's backups. The first
time a given drive is backed up, you're prompted for a friendly name for it
(e.g. "WD_Passport_2TB_01"); that name is then remembered (keyed by the
drive's volume serial number, in .drive_names.json next to this script) and
reused automatically on every later run, so you don't have to retype it.

Safe to run repeatedly: a file is (re)copied only if it's new, its size
differs, or the source is newer than the destination copy. Unchanged files
are skipped, so re-running doesn't waste time or trigger needless Dropbox
re-uploads.

Usage:

    python backup_to_dropbox.py
    # → first run for a drive: prompts for a friendly name, then backs up
    #   D:\\MultiWell_swim to .../Data_Backup/<friendly name>
    #   later runs on the same drive reuse the saved name automatically.

    python backup_to_dropbox.py --dry_run
    # → prints what would be copied/updated/skipped without touching anything

    python backup_to_dropbox.py --source "D:/MultiWell_swim" --dest "C:/.../Data_Backup" \\
        --drive_name WD_Passport_2TB_01 --video_exts .avi,.mp4
    # → explicitly names/renames the drive and updates the saved mapping
"""

import argparse
import ctypes
import json
import os
import re
import shutil
import sys

# Redirected/piped/logged output (e.g. Task Scheduler) falls back to the
# system codepage, which can't encode arrows/em-dashes below; force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = r"D:\MultiWell_swim"
DEFAULT_DEST = r"C:\Users\jl200\Dropbox\JHU_2026_spring\Multiwell_swim\Data_Backup"
DEFAULT_VIDEO_EXTS = ".avi,.mp4,.mov,.mkv,.wmv"
DRIVE_NAME_MAP_PATH = os.path.join(SCRIPT_DIR, ".drive_names.json")

# Two-second tolerance to absorb FAT32-style mtime rounding across drives.
MTIME_TOLERANCE_SEC = 2.0

_INVALID_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*]')


def _get_volume_info(path: str) -> tuple:
    """Return (volume_label_or_drive_letter, serial_number_or_None) for the
    drive containing `path`. Falls back to the bare drive letter (e.g. "D")
    if the drive is unlabeled or can't be queried (e.g. non-Windows).
    """
    drive = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
    fallback = drive.rstrip("\\:") or "drive"
    try:
        name_buf = ctypes.create_unicode_buffer(261)
        serial = ctypes.c_ulong(0)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(drive), name_buf,
            ctypes.sizeof(name_buf) // ctypes.sizeof(ctypes.c_wchar),
            ctypes.byref(serial), None, None, None, 0)
        if ok:
            return (name_buf.value or fallback, serial.value)
    except (AttributeError, OSError):
        pass
    return (fallback, None)


def _sanitize_folder_name(name: str) -> str:
    """Strip characters that aren't valid in a Windows folder name."""
    return _INVALID_FOLDER_CHARS.sub("_", name).strip() or "drive"


def _load_drive_name_map() -> dict:
    if os.path.exists(DRIVE_NAME_MAP_PATH):
        with open(DRIVE_NAME_MAP_PATH, "r") as fh:
            return json.load(fh)
    return {}


def _save_drive_name_map(name_map: dict) -> None:
    with open(DRIVE_NAME_MAP_PATH, "w") as fh:
        json.dump(name_map, fh, indent=2)


def _resolve_drive_name(source: str, explicit_name: str) -> str:
    """Determine the backup subfolder name for the drive containing `source`.

    Priority: explicit --drive_name > previously saved name for this drive's
    volume serial number > interactive prompt (saved for next time) > volume
    label/drive letter fallback if not running interactively.
    """
    label, serial = _get_volume_info(source)
    name_map = _load_drive_name_map()
    serial_key = str(serial) if serial is not None else None

    if explicit_name:
        drive_name = explicit_name
    elif serial_key and serial_key in name_map:
        drive_name = name_map[serial_key]
        print(f"[Backup] Using saved name '{drive_name}' for this drive (serial {serial_key}).")
    elif serial_key and sys.stdin.isatty():
        entered = input(
            f"No saved name for this drive (volume label '{label}', serial {serial_key}).\n"
            f"Enter a friendly name for its backup folder [{label}]: "
        ).strip()
        drive_name = entered or label
    else:
        print("[Backup] No saved name and not running interactively; "
              f"falling back to volume label '{label}'.")
        drive_name = label

    if serial_key:
        name_map[serial_key] = drive_name
        _save_drive_name_map(name_map)

    return _sanitize_folder_name(drive_name)


def _is_up_to_date(src_path: str, dst_path: str) -> bool:
    """Return True if dst_path already reflects src_path's current content."""
    if not os.path.exists(dst_path):
        return False
    src_stat = os.stat(src_path)
    dst_stat = os.stat(dst_path)
    if src_stat.st_size != dst_stat.st_size:
        return False
    return dst_stat.st_mtime >= src_stat.st_mtime - MTIME_TOLERANCE_SEC


def backup(source: str, dest: str, video_exts: set, dry_run: bool) -> None:
    n_copied = n_updated = n_skipped = n_skipped_video = n_errors = 0
    bytes_copied = 0

    for root, _dirs, files in os.walk(source):
        rel_dir = os.path.relpath(root, source)
        dest_dir = os.path.normpath(os.path.join(dest, rel_dir))

        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in video_exts:
                n_skipped_video += 1
                continue

            src_path = os.path.join(root, name)
            dst_path = os.path.join(dest_dir, name)
            rel_path = os.path.join(rel_dir, name)

            if _is_up_to_date(src_path, dst_path):
                n_skipped += 1
                continue

            is_update = os.path.exists(dst_path)
            action = "UPDATE" if is_update else "COPY"
            print(f"[{action}] {rel_path}")

            if not dry_run:
                try:
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                except OSError as exc:
                    print(f"[ERROR] {rel_path}: {exc}", file=sys.stderr)
                    n_errors += 1
                    continue

            bytes_copied += os.path.getsize(src_path)
            if is_update:
                n_updated += 1
            else:
                n_copied += 1

    print("\n[Backup] Summary" + (" (dry run — nothing was written)" if dry_run else ""))
    print(f"  New files copied   : {n_copied}")
    print(f"  Existing files updated : {n_updated}")
    print(f"  Unchanged, skipped : {n_skipped}")
    print(f"  Raw videos skipped : {n_skipped_video}")
    print(f"  Errors             : {n_errors}")
    print(f"  Total data transferred : {bytes_copied / (1024 ** 2):.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Back up processed Multiwell Swim data (excluding raw videos) "
                     "from the hard drive to Dropbox.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Root folder to back up from.")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help="Root folder to back up to (Dropbox folder). Each drive "
                             "gets its own subfolder under this path (see --drive_name).")
    parser.add_argument("--drive_name", default=None,
                        help="Name of the subfolder under --dest for this drive's backup. "
                             "If omitted, reuses the name saved from a previous run for this "
                             "drive, or prompts for one on first use (saved for next time).")
    parser.add_argument("--video_exts", default=DEFAULT_VIDEO_EXTS,
                        help="Comma-separated, case-insensitive list of raw video "
                             "file extensions to exclude from backup.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be copied/updated without writing anything.")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        raise SystemExit(f"Source folder not found: {args.source}")

    video_exts = {e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
                  for e in args.video_exts.split(",") if e.strip()}

    drive_name = _resolve_drive_name(args.source, args.drive_name)
    dest_root = os.path.join(args.dest, drive_name)
    print(f"[Backup] Drive: {drive_name}  →  {dest_root}")

    os.makedirs(dest_root, exist_ok=True)
    backup(args.source, dest_root, video_exts, args.dry_run)


if __name__ == "__main__":
    main()
