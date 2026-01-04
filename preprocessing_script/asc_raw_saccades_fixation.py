import re
import pandas as pd
from pathlib import Path
import numpy as np

# ================================
# CONFIG
# ================================

ASC_DIR = "/Users/sophie/Library/CloudStorage/Box-Box/DCL_ARCHIVE/Documents/Events/exp159_EyetrackingSEM/results/ascFiles"
OUTPUT_DIR = "event_level_outputs"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

FPS = 59.94006   # Sp
# ================================
# REGEX PATTERNS
# ================================

RE_TRIAL = re.compile(r"TRIALID\s+(\d+)")
RE_VIDEO = re.compile(r"videot\s+([\d\.]+mp4)")
RE_FRAME = re.compile(r"\*CRT\*(\d+)")     # Frame messages #There is no frame messages yet

RE_FIX = re.compile(
    r"EFIX\s+\w+\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d\.]+)\s+([-\d\.]+)"
)

RE_SACC = re.compile(
    r"ESACC\s+\w+\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
)

# ================================
# GLOBAL STORAGE
# ================================

all_fixations = []
all_saccades = []

# ================================
# PROCESS FILES
# ================================

asc_files = sorted(Path(ASC_DIR).glob("*.asc"))
print(f"\n Found {len(asc_files)} ASC files\n")

for asc_path in asc_files:

    sub_id = asc_path.stem
    print(f"\nProcessing: {sub_id}")

    fixations = []
    saccades = []

    current_trial = None
    current_movie = None
    frame_start_by_trial = {}

    with open(asc_path, "r", errors="ignore") as f:
        for line in f:

            # ---------- Trial ----------
            trial_match = RE_TRIAL.search(line)
            if trial_match:
                current_trial = int(trial_match.group(1))

            # ---------- Movie ----------
            video_match = RE_VIDEO.search(line)
            if video_match:
                current_movie = video_match.group(1)

            # ---------- Frame Start (Frame == 1) ----------
            frame_match = RE_FRAME.search(line)
            if frame_match and current_trial is not None:
                frame_num = int(frame_match.group(1))
                if frame_num == 1:
                    ms_match = re.search(r"(\d+)", line)
                    if ms_match:
                        frame_start_by_trial[current_trial] = int(ms_match.group(1))

            # ---------- FIXATIONS ----------
            fix_match = RE_FIX.search(line)
            if fix_match:
                try:
                    start = int(fix_match.group(1))
                    end   = int(fix_match.group(2))
                    dur   = int(fix_match.group(3))
                    x     = float(fix_match.group(4))
                    y     = float(fix_match.group(5))
                except ValueError:
                    continue  # skip corrupted row

                frame_start = frame_start_by_trial.get(current_trial, np.nan)
                ms_video = start - frame_start if not np.isnan(frame_start) else np.nan
                calc_frame = int(np.floor((ms_video / 1000) * FPS) + 1) if not np.isnan(ms_video) else np.nan

                fixations.append({
                    "subject": sub_id,
                    "trial": current_trial,
                    "movie": current_movie,
                    "frame_start_ms": frame_start,
                    "start_time_ms": start,
                    "end_time_ms": end,
                    "ms_video": ms_video,
                    "calc_frame": calc_frame,
                    "duration_ms": dur,
                    "x_px": x,
                    "y_px": y
                })

            # ---------- SACCADES ----------
            sacc_match = RE_SACC.search(line)
            if sacc_match:
                try:
                    start = int(sacc_match.group(1))
                    end   = int(sacc_match.group(2))
                    dur   = int(sacc_match.group(3))
                    sx    = float(sacc_match.group(4))
                    sy    = float(sacc_match.group(5))
                    ex    = float(sacc_match.group(6))
                    ey    = float(sacc_match.group(7))
                    amp_d = float(sacc_match.group(8))
                except ValueError:
                    continue  # skip corrupted row

                frame_start = frame_start_by_trial.get(current_trial, np.nan)
                ms_video = start - frame_start if not np.isnan(frame_start) else np.nan
                calc_frame = int(np.floor((ms_video / 1000) * FPS) + 1) if not np.isnan(ms_video) else np.nan
                amp_px = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5

                saccades.append({
                    "subject": sub_id,
                    "trial": current_trial,
                    "movie": current_movie,
                    "frame_start_ms": frame_start,
                    "start_time_ms": start,
                    "end_time_ms": end,
                    "ms_video": ms_video,
                    "calc_frame": calc_frame,
                    "duration_ms": dur,
                    "start_x_px": sx,
                    "start_y_px": sy,
                    "end_x_px": ex,
                    "end_y_px": ey,
                    "amp_deg": amp_d,
                    "amp_px": amp_px
                })

    # ================================
    # SAVE PER-SUBJECT
    # ================================

    fix_df = pd.DataFrame(fixations)
    sacc_df = pd.DataFrame(saccades)

    fix_df.to_csv(Path(OUTPUT_DIR) / f"{sub_id}_fixations.csv", index=False)
    sacc_df.to_csv(Path(OUTPUT_DIR) / f"{sub_id}_saccades.csv", index=False)

    print(f"  → saved {len(fix_df)} fixations | {len(sacc_df)} saccades")

    all_fixations.append(fix_df)
    all_saccades.append(sacc_df)

# ================================
# SAVE GROUP FILES
# ================================

if all_fixations:
    pd.concat(all_fixations, ignore_index=True).to_csv(
        Path(OUTPUT_DIR) / "ALL_fixations.csv", index=False)

if all_saccades:
    pd.concat(all_saccades, ignore_index=True).to_csv(
        Path(OUTPUT_DIR) / "ALL_saccades.csv", index=False)

print("\n ALL FILES PROCESSED WITH FRAME_START ALIGNMENT")
