import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

FIX_FILE   = "event_level_outputs/ALL_fixations.csv"
OUT_CSV    = "frame_level_eye_ISC.csv"

# Use the display size from your EyeLink setup (DISPLAY_COORDS 0 0 1279 719)
WIDTH  = 1280
HEIGHT = 720

GAUSS_SIGMA = 50.0   # px; adjust if you want broader/narrower maps
MIN_SUBJ_PER_FRAME = 2   # need at least 2 to compute ISC

# ============================================================
# GAUSSIAN & HEATMAP HELPERS
# ============================================================

def gaussian_mask(width, height, sigma=10.0, center=None, weight=1.0):
    """
    2D Gaussian mask over [0,width) x [0,height).
    center = (x, y) in pixel coords.
    weight scales the peak height.
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    xx, yy = np.meshgrid(x, y)

    if center is None or np.any(np.isnan(center)):
        return np.zeros((height, width), dtype=np.float32)

    cx, cy = center
    return weight * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (sigma ** 2))


def subject_heatmap(fix_df, width, height, sigma=GAUSS_SIGMA):
    """
    Build a *single subject* heatmap for a given movie × frame, summing
    Gaussians over that subject's fixations.
    """
    hm = np.zeros((height, width), dtype=np.float32)

    for _, row in fix_df.iterrows():
        cx = row["x_px"]
        cy = row["y_px"]

        # Skip clearly invalid coords
        if np.isnan(cx) or np.isnan(cy):
            continue
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            continue

        hm += gaussian_mask(width, height, sigma=sigma, center=(cx, cy), weight=1.0)

    # Normalize to sum 1 (so maps are comparable across subjects)
    total = hm.sum()
    if total > 0:
        hm /= total
    return hm


def corr_2d(a, b):
    """
    Pearson correlation between two 2D maps (flattened).
    Returns np.nan if variance is zero.
    """
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)

    x -= x.mean()
    y -= y.mean()
    num = np.dot(x, y)
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if den == 0:
        return np.nan
    return num / den

# ============================================================
# LOAD FIXATION DATA
# ============================================================

fix = pd.read_csv(FIX_FILE)

# Keep only rows with valid frame and coordinates
fix = fix.dropna(subset=["calc_frame", "x_px", "y_px", "movie", "subject"])
fix["calc_frame"] = fix["calc_frame"].astype(int)

print(f"Loaded {len(fix)} fixation rows")

# ============================================================
# COMPUTE FRAME-LEVEL EYE-ISC
# ============================================================

results = []

for movie, df_movie in fix.groupby("movie"):
    print(f"\nProcessing movie: {movie}")

    frames = np.sort(df_movie["calc_frame"].unique())

    for frame in frames:
        df_frame = df_movie[df_movie["calc_frame"] == frame]

        # Build per-subject heatmaps for this frame
        subj_heatmaps = []
        subj_ids = []

        for subj, df_sf in df_frame.groupby("subject"):
            hm = subject_heatmap(df_sf, WIDTH, HEIGHT, sigma=GAUSS_SIGMA)

            # If subject has no valid fixations, skip
            if hm.sum() == 0:
                continue

            subj_heatmaps.append(hm)
            subj_ids.append(subj)

        n_subj = len(subj_heatmaps)
        if n_subj < MIN_SUBJ_PER_FRAME:
            continue  # cannot compute ISC with < 2 subjects

        H = np.stack(subj_heatmaps, axis=0)   # shape: (N, H, W)
        overall_mean = H.mean(axis=0)

        isc_vals = []
        for i in range(n_subj):
            # Mean of all other subjects (leave-one-out)
            mean_excl = (overall_mean * n_subj - H[i]) / (n_subj - 1)
            isc_i = corr_2d(H[i], mean_excl)
            if not np.isnan(isc_i):
                isc_vals.append(isc_i)

        if len(isc_vals) == 0:
            continue

        isc_vals = np.array(isc_vals, dtype=float)

        results.append({
            "movie": movie,
            "frame": int(frame),
            "n_subjects": n_subj,
            "isc_mean": float(isc_vals.mean()),
            "isc_sd": float(isc_vals.std(ddof=1)) if len(isc_vals) > 1 else 0.0
        })

# ============================================================
# SAVE TO CSV
# ============================================================

isc_df = pd.DataFrame(results).sort_values(["movie", "frame"])
isc_df.to_csv(OUT_CSV, index=False)

print(f"\n✅ Saved frame-level eye-ISC to: {OUT_CSV}")
print(isc_df.head())
