import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG 
# this is the script that take the output fromt eh processing for the futer analysis to calculate the fixation dufrration dan station anmiples 
# ============================================================
FIX_PATH  = "event_level_outputs/ALL_fixations.csv"
SACC_PATH = "event_level_outputs/ALL_saccades.csv"

OUTPUT_DIR = Path("event_level_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUT_FILE = OUTPUT_DIR / "FRAME_LEVEL_EYE_METRICS.csv"

MIN_SUBJECTS_PER_FRAME = 0    # safety threshold

# ============================================================
# LOAD DATA
# ============================================================

print("\n Loading fixation and saccade files...")

fix  = pd.read_csv(FIX_PATH)
sacc = pd.read_csv(SACC_PATH)

print("Fix rows:", len(fix))
print("Sacc rows:", len(sacc))
print("Fix subjects:", fix["subject"].nunique())
print("Sacc subjects:", sacc["subject"].nunique())

# ============================================================
# HARD EXCLUSION OF BAD SUBJECTS (NO FRAME ALIGNMENT)
# ============================================================

print("\n Removing subjects with missing frame alignment...")

bad_fix_subs = set(
    fix.loc[fix["calc_frame"].isna() | fix["ms_video"].isna(), "subject"]
)

bad_sacc_subs = set(
    sacc.loc[sacc["calc_frame"].isna() | sacc["ms_video"].isna(), "subject"]
)

bad_subjects = bad_fix_subs.union(bad_sacc_subs)

print("Subjects removed:", bad_subjects)

fix_clean  = fix[~fix["subject"].isin(bad_subjects)].copy()
sacc_clean = sacc[~sacc["subject"].isin(bad_subjects)].copy()

print(" Subjects retained:", fix_clean["subject"].nunique())

# ============================================================
# REMOVE SPARSE FRAMES (PREVENT BOUNDARY BIAS)
# ============================================================

print("\nRemoving sparse frames...")

fix_counts = (
    fix_clean
    .groupby(["movie", "calc_frame"])
    .subject.nunique()
    .reset_index(name="n_subjects")
)

valid_frames = fix_counts.query("n_subjects >= @MIN_SUBJECTS_PER_FRAME")

fix_clean = fix_clean.merge(
    valid_frames[["movie", "calc_frame"]],
    on=["movie", "calc_frame"],
    how="inner"
)

sacc_clean = sacc_clean.merge(
    valid_frames[["movie", "calc_frame"]],
    on=["movie", "calc_frame"],
    how="inner"
)

# ============================================================
# FRAME-LEVEL FIXATION AGGREGATION
# ============================================================

print("\n✅ Aggregating fixations to frame level...")

fix_frame_avg = (
    fix_clean
    .groupby(["movie", "calc_frame"])
    .agg(
        mean_fix_dur_ms = ("duration_ms", "mean"),
        sd_fix_dur_ms   = ("duration_ms", "std"),
        n_fix           = ("duration_ms", "count"),
        mean_fix_x      = ("x_px", "mean"),
        mean_fix_y      = ("y_px", "mean"),
        n_subjects_fix  = ("subject", "nunique")
    )
    .reset_index()
)

# ============================================================
# FRAME-LEVEL SACCADE AGGREGATION
# ============================================================

print("\n✅ Aggregating saccades to frame level...")

sacc_frame_avg = (
    sacc_clean
    .groupby(["movie", "calc_frame"])
    .agg(
        mean_sacc_dur_ms = ("duration_ms", "mean"),
        sd_sacc_dur_ms   = ("duration_ms", "std"),
        mean_sacc_amp_px = ("amp_px", "mean"),
        sd_sacc_amp_px   = ("amp_px", "std"),
        mean_sacc_amp_deg = ("amp_deg", "mean"),
        n_sacc           = ("duration_ms", "count"),
        n_subjects_sacc  = ("subject", "nunique")
    )
    .reset_index()
)

# ============================================================
# MERGE FIXATION + SACCADE FRAME TABLES
# ============================================================

print("\n✅ Merging fixations + saccades...")

frame_level = fix_frame_avg.merge(
    sacc_frame_avg,
    on=["movie", "calc_frame"],
    how="outer",
    suffixes=("_fix", "_sacc")
)

# ============================================================
# SAVE OUTPUT
# ============================================================

frame_level.to_csv(OUT_FILE, index=False)

print("\n✅ FRAME-LEVEL DATASET SAVED TO:")
print(OUT_FILE)

# ============================================================
# SANITY CHECKS
# ============================================================

print("\n✅ Preview:")
print(frame_level.head())

print("\n✅ Frame counts per movie:")
print(frame_level.groupby("movie")["calc_frame"].max())

print("\n✅ Subjects contributing per movie (fixations):")
print(
    fix_clean.groupby("movie")["subject"].nunique()
)

print("\n✅ Subjects contributing per movie (saccades):")
print(
    sacc_clean.groupby("movie")["subject"].nunique()
)
