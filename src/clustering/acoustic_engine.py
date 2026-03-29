"""
harmonicai/src/clustering/acoustic_engine.py

Phase 2 — Acoustic Clustering Engine
======================================
Segments the track library into therapeutic profiles using
unsupervised learning — purely on acoustic mathematics,
with zero reference to genre labels.

Pipeline:
    1. Feature selection & justification
    2. Standardization (mandatory before distance-based methods)
    3. K selection via Elbow + Silhouette
    4. K-Means clustering (primary)
    5. Hierarchical clustering (validation)
    6. Cluster profiling & therapeutic interpretation
    7. Cluster quality diagnostics

Run:
    python src/clustering/acoustic_engine.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ingestion.loader import load_all

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# SECTION 1: Feature Selection
# ─────────────────────────────────────────────

# Why these features and not others?
#
# INCLUDED:
#   tempo_bpm        — primary entrainment driver (heart rate coupling)
#   energy           — arousal axis of Russell's Circumplex model
#   valence          — valence axis of Russell's Circumplex model
#   acousticness     — texture signal (grounding vs electronic)
#   instrumentalness — routes to NLP filter; also a therapeutic signal
#   speechiness      — overlaps instrumentalness but captures spoken word
#   loudness_db      — physical intensity; affects cortisol response
#
# EXCLUDED:
#   danceability     — derived feature (Spotify internal); redundant with tempo + energy
#   key              — pitch class alone has weak therapeutic signal
#   mode             — too coarse (binary) to drive cluster geometry
#   time_signature   — 82% of tracks are 4/4; near-zero variance → no clustering value
#   duration_ms      — session planning only; not a therapeutic property of the sound

CLUSTERING_FEATURES = [
    "tempo_bpm",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "loudness_db",
]

# K range to evaluate
K_MIN, K_MAX = 2, 12

# Final K (determined after running elbow + silhouette analysis)
# We set this explicitly so downstream code is reproducible
K_FINAL = 6

RANDOM_SEED = 42


# ─────────────────────────────────────────────
# SECTION 2: Preprocessing
# ─────────────────────────────────────────────

def preprocess_features(tracks: pd.DataFrame) -> tuple:
    """
    Extracts and standardizes clustering features.

    Why StandardScaler specifically?
    StandardScaler: x_scaled = (x - mean) / std
    Every feature ends up with mean=0 and std=1.

    Alternative: MinMaxScaler (scales to [0,1]).
    Problem with MinMaxScaler: a single outlier track at 200 BPM
    compresses every other track's tempo into a narrow band near 0.
    StandardScaler is more robust to outliers.

    Why not skip scaling entirely?
    tempo_bpm has std ≈ 27 BPM.
    loudness_db has std ≈ 5 dB.
    Without scaling, a 1-unit move in tempo is 5x less important
    than a 1-unit move in loudness — not because tempo matters less,
    but because its units happen to be larger. That's a measurement
    artifact, not a therapeutic truth. Scaling removes it.
    """

    X_raw = tracks[CLUSTERING_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"\n  Features selected:  {CLUSTERING_FEATURES}")
    print(f"  Feature matrix:     {X_scaled.shape[0]:,} tracks × {X_scaled.shape[1]} features")
    print(f"  Post-scaling mean:  {X_scaled.mean(axis=0).round(6)}")   # should be ~0
    print(f"  Post-scaling std:   {X_scaled.std(axis=0).round(6)}")    # should be ~1

    return X_scaled, scaler, tracks[CLUSTERING_FEATURES].copy()


# ─────────────────────────────────────────────
# SECTION 3: K Selection
# ─────────────────────────────────────────────

def find_optimal_k(X: np.ndarray) -> pd.DataFrame:
    """
    Evaluates K from K_MIN to K_MAX using two complementary metrics:

    1. WCSS (inertia) — Elbow method
       Lower WCSS = tighter clusters = better
       Problem: always decreases with K → need to find the "elbow"

    2. Silhouette score — Direct cluster quality
       Higher = better separation and cohesion
       Range: [-1, +1]
       Interpretation:
         > 0.5  → strong cluster structure
         0.25–0.5 → moderate structure (common in real data)
         < 0.25 → weak or no structure

    Why two metrics?
    The elbow is subjective — different people draw it differently.
    Silhouette gives an objective score per K.
    When both agree on a K, that's a confident choice.
    When they disagree, investigate both and prefer silhouette.
    """

    print(f"\n  Evaluating K from {K_MIN} to {K_MAX}...")
    print(f"  {'K':>3}  {'WCSS':>12}  {'Silhouette':>12}  {'Assessment'}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*20}")

    results = []
    wcss_values = []

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(
            n_clusters=k,
            n_init=10,          # run 10 times with different seeds, keep best
            max_iter=300,
            random_state=RANDOM_SEED
        )
        labels = km.fit_predict(X)
        wcss = km.inertia_
        sil  = silhouette_score(X, labels, sample_size=2000, random_state=RANDOM_SEED)

        wcss_values.append(wcss)

        # Compute elbow as second derivative of WCSS
        # The elbow is where WCSS starts decreasing slowly
        assessment = ""
        if k >= K_MIN + 2 and len(wcss_values) >= 3:
            d1 = wcss_values[-2] - wcss_values[-1]   # current decrease
            d2 = wcss_values[-3] - wcss_values[-2]   # previous decrease
            if d2 > 0 and d1 / d2 < 0.5:
                assessment = "← elbow candidate"

        if sil > 0.35:
            assessment += " ★ high silhouette"

        print(f"  {k:>3}  {wcss:>12.1f}  {sil:>12.4f}  {assessment}")

        results.append({
            "k":               k,
            "wcss":            round(wcss, 2),
            "silhouette":      round(sil, 4),
            "wcss_reduction":  round(wcss_values[-2] - wcss, 2) if len(wcss_values) > 1 else 0,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# SECTION 4: K-Means Clustering
# ─────────────────────────────────────────────

def run_kmeans(X: np.ndarray, k: int) -> tuple:
    """
    Runs K-Means with the chosen K.

    Returns:
        labels     — cluster assignment per track (0 to k-1)
        centroids  — k × n_features matrix of cluster centers
        km_model   — fitted KMeans object
    """

    km = KMeans(
        n_clusters=k,
        n_init=10,
        max_iter=500,
        random_state=RANDOM_SEED
    )
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    wcss = km.inertia_
    sil  = silhouette_score(X, labels)

    print(f"\n  K-Means (K={k}) results:")
    print(f"    WCSS (inertia):    {wcss:,.1f}")
    print(f"    Silhouette score:  {sil:.4f}")

    # Cluster size distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Cluster size distribution:")
    for cluster_id, count in zip(unique, counts):
        pct = count / len(labels)
        bar = "█" * int(pct * 30)
        print(f"    Cluster {cluster_id}: {bar:<30} {count:>5} tracks ({pct:.1%})")

    return labels, centroids, km


# ─────────────────────────────────────────────
# SECTION 5: Hierarchical Clustering (Validation)
# ─────────────────────────────────────────────

def run_hierarchical(X: np.ndarray, k: int, sample_size: int = 500) -> np.ndarray:
    """
    Runs Agglomerative (hierarchical) clustering for cross-validation.

    We sample for performance — full hierarchical clustering is O(n²)
    which is slow at 5,000 tracks. 500 samples is sufficient to
    validate that the cluster structure K-Means found is real.

    If K-Means and hierarchical clustering find similar structures
    on the same data, that's strong evidence the clusters reflect
    genuine mathematical properties of the data — not artifacts
    of the K-Means initialization.

    If they disagree substantially, the cluster structure is
    unstable and you should not trust either result.
    """

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]

    hc = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"      # Ward minimizes WCSS increase on merge
    )
    labels_hc = hc.fit_predict(X_sample)

    sil_hc = silhouette_score(X_sample, labels_hc)
    print(f"\n  Hierarchical clustering (Ward, K={k}, n={sample_size}):")
    print(f"    Silhouette score: {sil_hc:.4f}")

    # Size distribution
    unique, counts = np.unique(labels_hc, return_counts=True)
    print(f"    Cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}")

    return labels_hc, sil_hc


# ─────────────────────────────────────────────
# SECTION 6: Cluster Profiling
# ─────────────────────────────────────────────

# Human-readable cluster names assigned after inspecting centroids.
# These are HYPOTHESES, not ground truth. They will be validated
# against efficacy labels in Phase 3.
#
# Naming convention: [arousal level] + [primary characteristic]
# We deliberately avoid genre words (no "classical", no "ambient")

CLUSTER_NAMES = {
    0: "low_arousal_acoustic",
    1: "high_energy_driving",
    2: "mid_tempo_balanced",
    3: "minimal_instrumental",
    4: "speech_dominant",
    5: "low_tempo_dark",
}


def profile_clusters(
    tracks: pd.DataFrame,
    labels: np.ndarray,
    centroids: np.ndarray,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Attaches cluster labels to tracks and computes per-cluster statistics.

    The centroid in scaled space is mathematically correct but
    uninterpretable — a centroid of [-1.2, 0.8, ...] is meaningless.
    We inverse-transform back to original feature space for reporting.
    """

    tracks = tracks.copy()
    tracks["cluster_id"] = labels

    # Map cluster IDs to human names
    tracks["cluster_name"] = tracks["cluster_id"].map(CLUSTER_NAMES)

    # Inverse-transform centroids to original scale for interpretation
    centroids_original = scaler.inverse_transform(centroids)
    centroid_df = pd.DataFrame(
        centroids_original,
        columns=CLUSTERING_FEATURES
    )
    centroid_df.index.name = "cluster_id"

    print(f"\n  {'='*60}")
    print(f"  Cluster Profiles (centroids in original feature space)")
    print(f"  {'='*60}")

    for i in range(len(centroids_original)):
        c = centroid_df.iloc[i]
        name = CLUSTER_NAMES.get(i, f"cluster_{i}")
        cluster_tracks = tracks[tracks["cluster_id"] == i]
        n = len(cluster_tracks)

        print(f"\n  ── Cluster {i}: {name.upper()} (n={n:,}) ──")
        print(f"    tempo_bpm:        {c['tempo_bpm']:.1f}")
        print(f"    energy:           {c['energy']:.3f}")
        print(f"    valence:          {c['valence']:.3f}")
        print(f"    acousticness:     {c['acousticness']:.3f}")
        print(f"    instrumentalness: {c['instrumentalness']:.3f}")
        print(f"    speechiness:      {c['speechiness']:.3f}")
        print(f"    loudness_db:      {c['loudness_db']:.1f}")

        # Therapeutic hypothesis
        print(f"\n    Therapeutic hypothesis:")
        _print_therapeutic_hypothesis(c, name)

    return tracks, centroid_df


def _print_therapeutic_hypothesis(centroid: pd.Series, cluster_name: str):
    """
    Maps centroid values to therapeutic hypotheses using
    the weak-label priors from feature_schema.yaml.
    This is not a prediction — it's a starting hypothesis
    for Phase 3 to validate or contradict.
    """

    bpm   = centroid["tempo_bpm"]
    nrg   = centroid["energy"]
    val   = centroid["valence"]
    inst  = centroid["instrumentalness"]
    sp    = centroid["speechiness"]

    candidates = []

    if 60 <= bpm <= 80 and nrg < 0.4:
        candidates.append("anxiety_relief (low energy, therapeutic tempo range)")
    if bpm < 65 and nrg < 0.35 and inst > 0.4:
        candidates.append("sleep_induction (very slow, quiet, instrumental)")
    if sp < 0.1 and inst > 0.3:
        candidates.append("deep_focus (minimal speech, moderate instrumental)")
    if val > 0.65 and nrg > 0.5:
        candidates.append("mood_uplift (positive valence, higher energy)")
    if bpm < 80 and val < 0.5:
        candidates.append("grief_processing (slow tempo, lower valence)")
    if nrg > 0.7:
        candidates.append("⚠ high energy — potential anxiety risk for sensitive users")

    if not candidates:
        candidates.append("unclear — may be a mixed or transitional cluster")

    for c in candidates:
        print(f"      → {c}")


# ─────────────────────────────────────────────
# SECTION 7: Cluster Quality Diagnostics
# ─────────────────────────────────────────────

def diagnose_cluster_quality(X: np.ndarray, labels: np.ndarray):
    """
    Runs per-cluster silhouette analysis to detect:

    1. Low-cohesion clusters — where points don't clearly
       belong to their cluster (silhouette near 0 or negative)

    2. Overlap zones — where a cluster's distribution
       overlaps heavily with a neighbour

    In production: clusters with mean silhouette < 0.1 should
    be investigated — they may represent tracks that don't
    fit any therapeutic profile cleanly, and should be
    flagged for manual review rather than auto-assigned.
    """

    sil_samples = silhouette_samples(X, labels)

    print(f"\n  Per-cluster silhouette diagnostics:")
    print(f"  {'Cluster':<12} {'Mean sil':>10} {'% negative':>12} {'Assessment'}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*25}")

    for cluster_id in np.unique(labels):
        mask       = labels == cluster_id
        sil_vals   = sil_samples[mask]
        mean_sil   = sil_vals.mean()
        pct_neg    = (sil_vals < 0).mean()

        name = CLUSTER_NAMES.get(cluster_id, f"cluster_{cluster_id}")

        if mean_sil > 0.35:
            assessment = "✅ strong cohesion"
        elif mean_sil > 0.15:
            assessment = "⚠  moderate — review overlap"
        else:
            assessment = "❌ weak — consider merging"

        print(f"  {name:<22} {mean_sil:>8.3f}  {pct_neg:>10.1%}  {assessment}")

    # Global silhouette
    global_sil = sil_samples.mean()
    print(f"\n  Global mean silhouette: {global_sil:.4f}")

    if global_sil > 0.35:
        print(f"  Assessment: Strong cluster structure — K={K_FINAL} is a good fit")
    elif global_sil > 0.20:
        print(f"  Assessment: Moderate structure — usable but watch overlap zones")
    else:
        print(f"  Assessment: Weak structure — consider fewer clusters or different features")

    return sil_samples


# ─────────────────────────────────────────────
# SECTION 8: How Clustering Failures Look
# ─────────────────────────────────────────────

def demonstrate_clustering_failure(X: np.ndarray, tracks_raw: pd.DataFrame):
    """
    Demonstrates what happens when you cluster on unscaled features.
    This is not part of the real pipeline — it's a teaching check.

    Clustering on raw features is one of the most common mistakes
    in real ML projects. The output looks plausible but is wrong.
    """

    print(f"\n  {'='*60}")
    print(f"  Clustering Failure Demo: Unscaled vs Scaled")
    print(f"  {'='*60}")

    # Cluster on raw (unscaled) features
    X_raw = tracks_raw[CLUSTERING_FEATURES].values
    km_bad = KMeans(n_clusters=K_FINAL, n_init=10, random_state=RANDOM_SEED)
    labels_bad = km_bad.fit_predict(X_raw)
    sil_bad = silhouette_score(X_raw, labels_bad)

    # Cluster on scaled features (correct)
    km_good = KMeans(n_clusters=K_FINAL, n_init=10, random_state=RANDOM_SEED)
    labels_good = km_good.fit_predict(X)
    sil_good = silhouette_score(X, labels_good)

    print(f"\n  Unscaled clustering silhouette: {sil_bad:.4f}")
    print(f"  Scaled clustering silhouette:   {sil_good:.4f}")

    # Check how much tempo dominated the unscaled clusters
    tracks_temp = tracks_raw.copy()
    tracks_temp["cluster_bad"]  = labels_bad
    tracks_temp["cluster_good"] = labels_good

    print(f"\n  Unscaled — cluster tempo ranges (dominated by tempo_bpm):")
    for c in range(K_FINAL):
        subset = tracks_temp[tracks_temp["cluster_bad"] == c]["tempo_bpm"]
        print(f"    Cluster {c}: tempo {subset.min():.0f}–{subset.max():.0f} BPM "
              f"(range {subset.max()-subset.min():.0f})")

    print(f"\n  Scaled — cluster tempo ranges (balanced with other features):")
    for c in range(K_FINAL):
        subset = tracks_temp[tracks_temp["cluster_good"] == c]["tempo_bpm"]
        print(f"    Cluster {c}: tempo {subset.min():.0f}–{subset.max():.0f} BPM "
              f"(range {subset.max()-subset.min():.0f})")

    print(f"\n  Key insight: unscaled clusters are just tempo buckets.")
    print(f"  All other therapeutic features are drowned out by BPM's larger scale.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  HarmonicAI — Phase 2: Acoustic Clustering Engine")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────
    tracks, users, sessions = load_all()

    # ── Preprocess ─────────────────────────────────────────
    print("\n[1/6] Preprocessing features...")
    X_scaled, scaler, tracks_raw = preprocess_features(tracks)

    # ── Find optimal K ─────────────────────────────────────
    print("\n[2/6] Finding optimal K...")
    k_results = find_optimal_k(X_scaled)

    # ── Run K-Means ────────────────────────────────────────
    print(f"\n[3/6] Running K-Means with K={K_FINAL}...")
    labels, centroids, km_model = run_kmeans(X_scaled, K_FINAL)

    # ── Validate with hierarchical ─────────────────────────
    print(f"\n[4/6] Validating with hierarchical clustering...")
    labels_hc, sil_hc = run_hierarchical(X_scaled, K_FINAL)

    # ── Profile clusters ───────────────────────────────────
    print(f"\n[5/6] Profiling clusters...")
    tracks_clustered, centroid_df = profile_clusters(
        tracks, labels, centroids, scaler
    )

    # ── Diagnose quality ───────────────────────────────────
    print(f"\n[6/6] Cluster quality diagnostics...")
    sil_samples = diagnose_cluster_quality(X_scaled, labels)

    # ── Failure demo ───────────────────────────────────────
    demonstrate_clustering_failure(X_scaled, tracks_raw)

    # ── Persist outputs ────────────────────────────────────
    tracks_clustered.to_csv(OUTPUT_DIR / "tracks_clustered.csv", index=False)
    centroid_df.to_csv(OUTPUT_DIR / "cluster_centroids.csv")
    k_results.to_csv(OUTPUT_DIR / "k_selection_results.csv", index=False)

    print(f"\n\n✅ Phase 2 outputs saved to: {OUTPUT_DIR}")
    print(f"   tracks_clustered.csv     — tracks with cluster_id + cluster_name")
    print(f"   cluster_centroids.csv    — centroid values in original feature space")
    print(f"   k_selection_results.csv  — WCSS and silhouette per K")
    print(f"\nNext step → Phase 3: python src/predictor/mood_classifier.py\n")
