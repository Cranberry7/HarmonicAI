"""
harmonicai/src/ingestion/profiler.py

Phase 1 — User Baseline Profiler & Mood Normalizer
====================================================
Builds a statistical profile of each user from their sessions.
Normalizes mood scores to account for individual baseline differences.

The core problem this solves:
  User A reports mood 70/100 → "good day for them"
  User B reports mood 70/100 → "terrible day for them"

  If User A's baseline is 65 and User B's baseline is 85,
  the same raw score means completely different things.
  Raw scores are MEANINGLESS without normalization.

This is called person-mean centering — standard practice in
longitudinal psychological research.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ingestion.loader import load_all

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# SECTION 1: Mood Score Normalization
# ─────────────────────────────────────────────

def normalize_mood_scores(sessions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Applies person-mean centering to mood scores.

    Standard normalization (z-score across all users) is WRONG here.
    Why? Because it assumes all users have the same baseline.
    Person-mean centering computes each user's personal mean and
    expresses every score as a deviation from that personal mean.

    Formula:
        mood_centered = mood_raw - user_mean_mood

    After centering:
        +10 means "10 points above this user's personal average"
        -10 means "10 points below this user's personal average"

    This is the correct unit of therapeutic change.
    """

    # Compute per-user mood statistics from their session history
    user_mood_stats = (
        sessions.groupby("user_id")["mood_pre"]
        .agg(
            user_mood_mean="mean",
            user_mood_std="std",
            user_mood_median="median",
            user_session_count="count"
        )
        .reset_index()
    )

    # Fill std=NaN for users with only 1 session (std undefined)
    user_mood_stats["user_mood_std"] = user_mood_stats["user_mood_std"].fillna(0)

    # Merge stats back onto sessions
    sessions = sessions.merge(user_mood_stats, on="user_id", how="left")

    # Person-mean centering
    sessions["mood_pre_centered"]  = sessions["mood_pre"]  - sessions["user_mood_mean"]
    sessions["mood_post_centered"] = sessions["mood_post"] - sessions["user_mood_mean"]
    sessions["delta_mood"]         = sessions["mood_post"] - sessions["mood_pre"]

    # Z-score normalization per user (for users with >1 session and std > 0)
    # This expresses mood in units of "standard deviations from personal mean"
    has_variance = sessions["user_mood_std"] > 0
    sessions.loc[has_variance, "mood_pre_zscore"] = (
        sessions.loc[has_variance, "mood_pre_centered"] /
        sessions.loc[has_variance, "user_mood_std"]
    )
    sessions["mood_pre_zscore"] = sessions["mood_pre_zscore"].fillna(0)

    return sessions, user_mood_stats


# ─────────────────────────────────────────────
# SECTION 2: User Baseline Profile Builder
# ─────────────────────────────────────────────

def build_user_profiles(sessions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches user profiles with behavioral statistics derived
    from their session history.

    These derived features become inputs to the recommendation
    engine in Phase 3 — they capture who the user IS behaviorally,
    not just what they reported at onboarding.
    """

    # Per-user session-level aggregations
    session_agg = sessions.groupby("user_id").agg(
        total_sessions        = ("session_id",       "count"),
        completion_rate       = ("completed",         "mean"),
        mean_efficacy_rating  = ("efficacy_rating",   "mean"),
        std_efficacy_rating   = ("efficacy_rating",   "std"),
        mean_delta_mood       = ("delta_mood",         "mean"),
        positive_session_rate = ("efficacy_label",
                                  lambda x: (x == 1).mean()),
        negative_session_rate = ("efficacy_label",
                                  lambda x: (x == -1).mean()),
        most_common_intent    = ("session_intent",
                                  lambda x: x.mode()[0] if len(x) > 0 else None),
    ).reset_index()

    session_agg["std_efficacy_rating"] = session_agg["std_efficacy_rating"].fillna(0)

    # Merge with onboarding profile
    enriched = users.merge(session_agg, on="user_id", how="left")

    # Fill NaN for users with no sessions yet (cold start)
    fill_zero = ["total_sessions", "completion_rate", "mean_efficacy_rating",
                 "mean_delta_mood", "positive_session_rate", "negative_session_rate"]
    enriched[fill_zero] = enriched[fill_zero].fillna(0)

    # ── Engagement tier segmentation ────────────────────────
    # Segment users by engagement depth — useful for Phase 7
    # cold start mitigation
    def engagement_tier(row):
        if row["total_sessions"] == 0:
            return "cold_start"
        elif row["total_sessions"] < 5:
            return "new"
        elif row["total_sessions"] < 20:
            return "engaged"
        else:
            return "loyal"

    enriched["engagement_tier"] = enriched.apply(engagement_tier, axis=1)

    return enriched


# ─────────────────────────────────────────────
# SECTION 3: Descriptive Statistics Report
# ─────────────────────────────────────────────

def describe_user_population(enriched_users: pd.DataFrame, sessions: pd.DataFrame):
    """
    Prints a structured descriptive statistics report.

    This is the output a clinical data scientist would review
    before approving the dataset for model training.
    """

    print("\n" + "=" * 60)
    print("  PHASE 1 — User Population Report")
    print("=" * 60)

    # ── Population overview ────────────────────────────────
    print(f"\n📊 Population Overview")
    print(f"   Total users:          {len(enriched_users):,}")
    print(f"   Total sessions:       {len(sessions):,}")
    print(f"   Avg sessions/user:    {sessions.groupby('user_id').size().mean():.1f}")
    print(f"   Users with sensitivity flag: "
          f"{enriched_users['has_sensitivity_flag'].sum()} "
          f"({enriched_users['has_sensitivity_flag'].mean():.1%})")

    # ── Baseline mood distribution ──────────────────────────
    print(f"\n🧠 Baseline Mood Distribution (onboarding)")
    bm = enriched_users["baseline_mood"]
    print(f"   Mean:     {bm.mean():.1f}")
    print(f"   Median:   {bm.median():.1f}")
    print(f"   Std Dev:  {bm.std():.1f}")
    print(f"   IQR:      {bm.quantile(0.25):.1f} – {bm.quantile(0.75):.1f}")
    print(f"   Skewness: {bm.skew():.2f}  "
          f"{'(right-skewed)' if bm.skew() > 0.5 else '(left-skewed)' if bm.skew() < -0.5 else '(approx. symmetric)'}")

    # ── Selection bias evidence ─────────────────────────────
    print(f"\n⚠️  Selection Bias Check")
    print(f"   Users with baseline_mood < 50: "
          f"{(enriched_users['baseline_mood'] < 50).mean():.1%}  ← expected to be high (struggling users)")
    print(f"   Users with baseline_mood > 75: "
          f"{(enriched_users['baseline_mood'] > 75).mean():.1%}  ← expected to be low")

    # ── Intent distribution ─────────────────────────────────
    print(f"\n🎯 Primary Intent Distribution")
    intent_counts = enriched_users["primary_intent"].value_counts()
    for intent, count in intent_counts.items():
        pct = count / len(enriched_users)
        bar = "█" * int(pct * 40)
        print(f"   {intent:<20} {bar:<40} {pct:.1%}")

    # ── Age distribution ────────────────────────────────────
    print(f"\n👥 Age Band Distribution")
    for band, count in enriched_users["age_band"].value_counts().sort_index().items():
        pct = count / len(enriched_users)
        print(f"   {band:<10} {'█' * int(pct * 30):<30} {pct:.1%}")

    # ── Engagement tiers ────────────────────────────────────
    print(f"\n🔄 Engagement Tiers")
    for tier, count in enriched_users["engagement_tier"].value_counts().items():
        pct = count / len(enriched_users)
        print(f"   {tier:<15} {'█' * int(pct * 30):<30} {pct:.1%}")

    # ── Therapeutic outcomes ────────────────────────────────
    print(f"\n📈 Therapeutic Outcome Summary (across all sessions)")
    delta = sessions["delta_mood"]
    print(f"   Mean mood change (delta):  {delta.mean():+.2f} points")
    print(f"   Median mood change:        {delta.median():+.2f} points")
    print(f"   Std of mood change:        {delta.std():.2f}")
    print(f"   Sessions with improvement: {(delta > 5).mean():.1%}")
    print(f"   Sessions with worsening:   {(delta < -5).mean():.1%}")

    # ── Efficacy rating stats ───────────────────────────────
    print(f"\n⭐ User Efficacy Ratings")
    er = sessions["efficacy_rating"]
    print(f"   Mean:    {er.mean():.2f} / 10")
    print(f"   Median:  {er.median():.1f} / 10")
    print(f"   Std Dev: {er.std():.2f}")
    print(f"   Ratings ≥ 7 (positive): {(er >= 7).mean():.1%}")
    print(f"   Ratings ≤ 3 (negative): {(er <= 3).mean():.1%}")


# ─────────────────────────────────────────────
# SECTION 4: Stratified Correlation Analysis
# ─────────────────────────────────────────────

def analyze_feature_correlations(sessions: pd.DataFrame, tracks: pd.DataFrame):
    """
    Computes Pearson correlations between audio features and
    efficacy_label, STRATIFIED by session_intent.

    Why stratification is non-negotiable:
        A high-energy track: helpful for mood_uplift, harmful for anxiety_relief
        Computing correlation across both groups → signal cancels → r ≈ 0
        Computing separately → reveals true directional effect

    This is called Simpson's Paradox in the wild.
    """

    print("\n" + "=" * 60)
    print("  PHASE 1 — Stratified Correlation Analysis")
    print("=" * 60)

    # Join sessions with track features
    merged = sessions.merge(tracks, on="track_id", how="left")

    audio_features = [
        "tempo_bpm", "valence", "energy", "acousticness",
        "instrumentalness", "danceability", "speechiness", "loudness_db"
    ]

    intents = sorted(merged["session_intent"].unique())

    # Build correlation table
    corr_rows = []
    for intent in intents:
        subset = merged[merged["session_intent"] == intent]
        row = {"intent": intent, "n_sessions": len(subset)}
        for feat in audio_features:
            r = subset[feat].corr(subset["efficacy_label"])
            row[feat] = round(r, 3)
        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows).set_index("intent")

    print(f"\n📐 Pearson r: audio feature vs efficacy_label (per intent)")
    print(f"   (range: -1.0 to +1.0  |  >|0.1| = weak,  >|0.3| = moderate)\n")

    # Print formatted table
    feat_cols = audio_features
    header = f"{'Intent':<22} {'n':>5} " + " ".join(f"{f[:8]:>9}" for f in feat_cols)
    print("   " + header)
    print("   " + "-" * len(header))

    for intent in intents:
        row = corr_df.loc[intent]
        n = int(row["n_sessions"])
        vals = " ".join(
            f"{row[f]:>+9.3f}" for f in feat_cols
        )
        print(f"   {intent:<22} {n:>5} {vals}")

    print(f"\n   💡 Key insight: same feature, different sign across intents")
    print(f"      = evidence that stratification is essential")
    print(f"      = a single global model CANNOT capture this")

    return corr_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Load validated data
    tracks, users, sessions = load_all()

    print("\n[1/4] Normalizing mood scores...")
    sessions, user_mood_stats = normalize_mood_scores(sessions, users)

    print("\n[2/4] Building enriched user profiles...")
    enriched_users = build_user_profiles(sessions, users)

    print("\n[3/4] Generating population report...")
    describe_user_population(enriched_users, sessions)

    print("\n[4/4] Running stratified correlation analysis...")
    corr_df = analyze_feature_correlations(sessions, tracks)

    # ── Persist processed outputs ──────────────────────────
    sessions.to_csv(OUTPUT_DIR / "sessions_normalized.csv", index=False)
    enriched_users.to_csv(OUTPUT_DIR / "users_enriched.csv", index=False)
    corr_df.to_csv(OUTPUT_DIR / "feature_correlations.csv")

    print(f"\n\n✅ Phase 1 outputs saved to: {OUTPUT_DIR}")
    print(f"   sessions_normalized.csv  — mood scores + delta + z-score")
    print(f"   users_enriched.csv       — behavioral profiles")
    print(f"   feature_correlations.csv — stratified Pearson r table")
    print(f"\nNext step → Phase 2: python src/clustering/acoustic_engine.py\n")
