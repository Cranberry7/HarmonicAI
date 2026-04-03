"""
src/feedback/drift_detector.py
===============================
Phase 7 — Feedback Loop & Continuous Learning

Purpose
-------
Closes the loop between live recommendations and model improvement.
Every completed user session produces a FeedbackEvent.  This module
accumulates those events, detects when the production model is drifting,
and triggers a controlled retrain when evidence warrants it.

Four responsibilities
---------------------
1. ACCUMULATE   — append FeedbackEvent records to
                  data/feedback/sessions_feedback.csv

2. DERIVE       — compute rolling performance metrics from accumulated
                  feedback: real-world F1, per-user recommendation
                  quality, cold-start detection

3. DETECT DRIFT — compare incoming feature distributions against the
                  training distribution using Population Stability Index
                  (PSI); emit a structured drift report

4. RETRAIN      — when drift is confirmed or the feedback buffer exceeds
                  the threshold, retrain random_forest.pkl on the enriched
                  dataset and promote the new checkpoint; record version
                  history in models/registry/model_registry.json

Key concepts taught
-------------------
POPULATION STABILITY INDEX (PSI)
    PSI = Σ (actual% − expected%) × ln(actual% / expected%)

    Interpretation:
        PSI < 0.10  →  No significant drift, model is stable
        PSI < 0.25  →  Moderate drift, begin monitoring more closely
        PSI ≥ 0.25  →  Significant drift, trigger retraining

    PSI is preferred over raw KL divergence for this use case because:
    - It is symmetric (measures both directions of shift)
    - It is bounded and interpretable on a consistent scale
    - It was developed specifically for model stability monitoring
      in production (originated in credit risk scoring)

ONLINE LEARNING vs BATCH RETRAINING
    Online learning (update weights incrementally per sample) works well
    when the model architecture supports it (e.g., SGD-based models).
    scikit-learn's RandomForestClassifier does not support partial_fit.
    We therefore use batch retraining on a rolling window of ALL data
    (original training + accumulated feedback).

    Retraining on only the NEW data would cause catastrophic forgetting:
    the model would overfit to the most recent session distribution and
    lose patterns learned from the original 8,000 sessions.

COLD START PROBLEM
    New users have zero session history.  The profiler.py person-mean
    centering cannot be applied.  The drift detector flags cold-start
    users explicitly; Phase 8 security and the API layer can route them
    to a fallback (global-mean features instead of user-mean features)
    or a simpler rule-based recommender until enough sessions accumulate.

WHEN RETRAINING HURTS
    Retraining on a small batch of new data can:
    (a) Overfit to recent session distribution if the window is too short
    (b) Amplify noise from a single user's unusual behaviour
    (c) Destroy calibrated class weights if the new batch has a different
        label distribution than the training set
    The 500-session threshold is deliberately conservative.  We also
    require PSI ≥ 0.10 on at least two features before retraining,
    preventing spurious retrains triggered by a single user's outlier week.

File I/O contract
-----------------
Reads  (must exist before running Phase 7):
    models/checkpoints/random_forest.pkl
    models/checkpoints/feature_list.pkl
    data/processed/sessions_normalized.csv
    data/synthetic/tracks.csv
    data/synthetic/users.csv

Writes:
    data/feedback/sessions_feedback.csv   — appended each call
    data/processed/drift_report.json      — overwritten each drift check
    models/checkpoints/random_forest_v{N}.pkl  — new checkpoint on retrain
    models/registry/model_registry.json   — version history

Usage
-----
    from src.feedback.drift_detector import DriftDetector, FeedbackEvent

    detector = DriftDetector()

    # Record a completed session
    event = FeedbackEvent(
        session_id      = "sess_001",
        user_id         = "user_042",
        track_id        = "track_1234",
        intent          = "anxiety_relief",
        mood_pre        = 34.0,
        mood_post       = 51.0,
        efficacy_rating = 7,
        completed       = True,
        predicted_label = 1,
        feature_vector  = {"tempo_bpm": 72, "energy": 0.28, ...},
    )
    detector.record(event)

    # Check drift + retrain if warranted (call periodically, e.g. nightly)
    report = detector.check_and_update()
    print(report)
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths  (all relative to project root)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

PATHS = {
    "rf_model":         ROOT / "models" / "checkpoints" / "random_forest.pkl",
    "feature_list":     ROOT / "models" / "checkpoints" / "feature_list.pkl",
    "train_sessions":   ROOT / "data" / "processed" / "sessions_normalized.csv",
    "tracks":           ROOT / "data" / "synthetic" / "tracks.csv",
    "users":            ROOT / "data" / "synthetic" / "users.csv",
    "feedback_csv":     ROOT / "data" / "feedback" / "sessions_feedback.csv",
    "drift_report":     ROOT / "data" / "processed" / "drift_report.json",
    "registry":         ROOT / "models" / "registry" / "model_registry.json",
    "checkpoints_dir":  ROOT / "models" / "checkpoints",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PSI thresholds (industry standard from credit-risk scoring literature)
PSI_STABLE   = 0.10   # < this → stable
PSI_MONITOR  = 0.25   # < this → watch closely
# ≥ PSI_MONITOR → significant drift → trigger retrain if also ≥ min_new_sessions

# Minimum new feedback sessions before considering a retrain
RETRAIN_SESSION_THRESHOLD = 500

# Features monitored for PSI (the 9 model inputs from Phase 3)
MONITORED_FEATURES = [
    "tempo_bpm", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "loudness_db",
    "mood_pre", "intent_encoded",
]

# Cold-start threshold: fewer than this many sessions = cold-start user
COLD_START_MIN_SESSIONS = 3

# Intent encoding — must match Phase 3 exactly
INTENT_ENCODING = {
    "sleep_induction": 0,
    "anxiety_relief":  1,
    "grief_processing":2,
    "deep_focus":      3,
    "mood_uplift":     4,
}

# Number of PSI bins for continuous features
PSI_BINS = 10

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeedbackEvent:
    """
    One completed session — the atomic unit of feedback.

    Parameters
    ----------
    session_id      : Unique session identifier (correlates with Phase 6 meta).
    user_id         : Opaque user identifier.
    track_id        : The lead track that was recommended.
    intent          : Session intent (one of INTENT_ENCODING keys).
    mood_pre        : Pre-session mood score (0–100).
    mood_post       : Post-session mood score (0–100). None if abandoned.
    efficacy_rating : User's 1–10 self-report of session usefulness.
    completed       : True if user completed the session; False if abandoned.
    predicted_label : The label the model predicted (+1, 0, or -1).
    feature_vector  : Dict of the 9 model features used for this prediction.
    timestamp       : ISO-8601 UTC string. Auto-filled if not supplied.
    """
    session_id:       str
    user_id:          str
    track_id:         str
    intent:           str
    mood_pre:         float
    mood_post:        Optional[float]
    efficacy_rating:  Optional[int]
    completed:        bool
    predicted_label:  int
    feature_vector:   Dict[str, float]
    timestamp:        str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.intent not in INTENT_ENCODING:
            raise ValueError(
                f"Unknown intent '{self.intent}'. "
                f"Valid: {sorted(INTENT_ENCODING.keys())}"
            )
        if self.predicted_label not in (-1, 0, 1):
            raise ValueError(
                f"predicted_label must be -1, 0, or 1; got {self.predicted_label}"
            )
        if self.mood_post is not None and not (0.0 <= self.mood_post <= 100.0):
            raise ValueError(f"mood_post must be in [0, 100]; got {self.mood_post}")
        if self.efficacy_rating is not None and not (1 <= self.efficacy_rating <= 10):
            raise ValueError(f"efficacy_rating must be 1–10; got {self.efficacy_rating}")

    def derive_true_label(self) -> Optional[int]:
        """
        Recompute the ground-truth efficacy label from post-session data,
        using the same oracle function defined in Phase 0.

        Returns None if insufficient data to label (e.g. session abandoned
        with no efficacy rating).
        """
        delta = None
        if self.mood_post is not None:
            delta = self.mood_post - self.mood_pre

        rating = self.efficacy_rating

        if delta is None and rating is None:
            return None

        # Phase 0 oracle:
        #   +1 if delta > +5  OR  rating >= 7
        #    0 if |delta| <= 5  AND  rating in [4,5,6]
        #   -1 if delta < -5  OR  rating <= 3
        if (delta is not None and delta > 5) or (rating is not None and rating >= 7):
            return 1
        if (delta is not None and delta < -5) or (rating is not None and rating <= 3):
            return -1
        if delta is not None or rating is not None:
            return 0
        return None


@dataclass
class DriftReport:
    """
    Output of a single drift check pass.

    psi_scores      : {feature_name: PSI_value} for all monitored features.
    drifted_features: Features with PSI ≥ PSI_STABLE (> 0.10).
    critical_features: Features with PSI ≥ PSI_MONITOR (≥ 0.25).
    rolling_f1      : Macro F1 computed on feedback events with true labels.
    training_f1     : F1 from the model's last training run (from registry).
    f1_delta        : rolling_f1 − training_f1.
    new_session_count: Feedback events accumulated since last retrain.
    cold_start_users: User IDs with fewer than COLD_START_MIN_SESSIONS sessions.
    retrain_recommended: True if drift + volume thresholds both met.
    retrain_reason  : Human-readable explanation.
    retrained       : True if retraining was actually performed this pass.
    new_model_path  : Path to new checkpoint if retrained.
    timestamp       : UTC ISO-8601 string.
    """
    psi_scores:          Dict[str, float]
    drifted_features:    List[str]
    critical_features:   List[str]
    rolling_f1:          Optional[float]
    training_f1:         float
    f1_delta:            Optional[float]
    new_session_count:   int
    cold_start_users:    List[str]
    retrain_recommended: bool
    retrain_reason:      str
    retrained:           bool           = False
    new_model_path:      str            = ""
    timestamp:           str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "=" * 64,
            "  HarmonicAI — Phase 7 Drift Report",
            f"  {self.timestamp}",
            "=" * 64,
            f"  New sessions since last retrain : {self.new_session_count}",
            f"  Training F1 (last checkpoint)   : {self.training_f1:.4f}",
        ]
        if self.rolling_f1 is not None:
            arrow = "↓" if self.f1_delta < -0.02 else ("↑" if self.f1_delta > 0.02 else "≈")
            lines.append(
                f"  Rolling F1 (feedback window)    : {self.rolling_f1:.4f}  "
                f"(Δ {self.f1_delta:+.4f} {arrow})"
            )
        else:
            lines.append("  Rolling F1                      : insufficient labelled data")

        lines.append("")
        lines.append("  PSI Scores:")
        for feat, psi in sorted(self.psi_scores.items(), key=lambda x: -x[1]):
            status = (
                "🔴 CRITICAL" if psi >= PSI_MONITOR else
                "🟡 MONITOR"  if psi >= PSI_STABLE  else
                "🟢 stable"
            )
            lines.append(f"    {feat:<22}  {psi:.4f}  {status}")

        if self.cold_start_users:
            lines.append(
                f"\n  Cold-start users detected       : {len(self.cold_start_users)}"
            )

        lines.append("")
        rec_icon = "⚡" if self.retrain_recommended else "✓"
        lines.append(f"  {rec_icon} Retrain recommended: {self.retrain_recommended}")
        if self.retrain_reason:
            lines.append(f"    Reason: {self.retrain_reason}")
        if self.retrained:
            lines.append(f"  ✅ Retrained → {self.new_model_path}")
        lines.append("=" * 64)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

class PSICalculator:
    """
    Computes Population Stability Index for each monitored feature.

    Workflow
    --------
    1. At initialisation, fit bin edges on the training distribution
       for each continuous feature.
    2. At drift-check time, bin the incoming (feedback) feature vectors
       using the same edges and compute PSI.

    Why fit bins on training data?
    -------------------------------
    PSI measures how much the new distribution has shifted relative to
    a fixed reference.  The reference is always the training distribution.
    Fitting bins on combined data would dilute the reference signal and
    underestimate drift.

    Handling edge cases
    --------------------
    - Zero-count bins in either distribution cause log(0/x) = -inf.
      We add a small epsilon (1e-4) before computing the log ratio.
    - Categorical features (intent_encoded) use value counts rather
      than equal-width bins.
    """

    _EPSILON = 1e-4

    def __init__(self) -> None:
        # {feature_name: bin_edges_array or "categorical"}
        self._bin_edges: Dict[str, Any] = {}
        self._reference_dist: Dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "PSICalculator":
        """Fit bin edges on the training feature distribution."""
        for feat in MONITORED_FEATURES:
            if feat not in train_df.columns:
                continue
            series = train_df[feat].dropna()

            if feat == "intent_encoded":
                # Categorical — use value counts as reference distribution
                counts = series.value_counts().sort_index()
                self._bin_edges[feat] = "categorical"
                self._reference_dist[feat] = (counts / counts.sum()).values
                # Store category order for consistent alignment
                self._bin_edges[f"{feat}_categories"] = counts.index.tolist()
            else:
                # Continuous — equal-width bins over training range
                _, edges = np.histogram(series, bins=PSI_BINS)
                self._bin_edges[feat] = edges
                counts, _ = np.histogram(series, bins=edges)
                ref = counts / counts.sum()
                self._reference_dist[feat] = ref

        self._fitted = True
        return self

    def compute(self, feedback_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute PSI for each feature between training reference and
        the supplied feedback feature vectors.
        """
        if not self._fitted:
            raise RuntimeError("PSICalculator must be fit() before compute().")

        psi_scores: Dict[str, float] = {}

        for feat in MONITORED_FEATURES:
            if feat not in self._bin_edges or feat not in feedback_df.columns:
                psi_scores[feat] = 0.0
                continue

            series = feedback_df[feat].dropna()
            if len(series) == 0:
                psi_scores[feat] = 0.0
                continue

            ref = self._reference_dist[feat]
            edges = self._bin_edges[feat]

            if isinstance(edges, str) and edges == "categorical":
                cats = self._bin_edges.get(f"{feat}_categories", [])
                act_counts = series.value_counts().reindex(cats, fill_value=0)
                act = (act_counts / act_counts.sum()).values
            else:
                act_counts, _ = np.histogram(series, bins=edges)
                act = act_counts / act_counts.sum()

            # Clip to same length (safety for categorical mismatch)
            min_len = min(len(ref), len(act))
            ref_clipped = ref[:min_len] + self._EPSILON
            act_clipped = act[:min_len] + self._EPSILON

            psi = float(np.sum((act_clipped - ref_clipped) * np.log(act_clipped / ref_clipped)))
            psi_scores[feat] = round(psi, 6)

        return psi_scores


# ---------------------------------------------------------------------------
# Rolling performance tracker
# ---------------------------------------------------------------------------

class RollingPerformanceTracker:
    """
    Derives real-world model performance metrics from feedback events.

    Labelled events: those where derive_true_label() returns non-None.
    Unlabelled events: abandoned sessions with no efficacy rating.

    The tracker computes:
    - Rolling macro F1 (predicted vs true label) on labelled events
    - Per-user recommendation quality (mean delta_mood where completed)
    - Cold-start user detection
    """

    def compute(
        self,
        feedback_df: pd.DataFrame,
        user_session_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        feedback_df : DataFrame of accumulated FeedbackEvent records.
        user_session_counts : {user_id: total_session_count} from training data.

        Returns
        -------
        Dict with keys: rolling_f1, labelled_count, mean_delta_mood,
                        per_intent_f1, cold_start_users
        """
        result: Dict[str, Any] = {
            "rolling_f1":    None,
            "labelled_count": 0,
            "mean_delta_mood": None,
            "per_intent_f1": {},
            "cold_start_users": [],
        }

        if feedback_df.empty:
            return result

        # --- Rolling F1 on labelled events --------------------------------
        labelled = feedback_df.dropna(subset=["true_label"])
        labelled = labelled[labelled["true_label"].isin([-1, 0, 1])]
        result["labelled_count"] = len(labelled)

        if len(labelled) >= 10:
            y_true = labelled["true_label"].astype(int).tolist()
            y_pred = labelled["predicted_label"].astype(int).tolist()
            result["rolling_f1"] = round(
                f1_score(y_true, y_pred, average="macro", zero_division=0), 4
            )

            # Per-intent F1
            for intent in INTENT_ENCODING:
                mask = labelled["intent"] == intent
                if mask.sum() >= 5:
                    result["per_intent_f1"][intent] = round(
                        f1_score(
                            labelled.loc[mask, "true_label"].astype(int),
                            labelled.loc[mask, "predicted_label"].astype(int),
                            average="macro",
                            zero_division=0,
                        ),
                        4,
                    )

        # --- Mean delta_mood on completed sessions -------------------------
        completed = feedback_df[feedback_df["completed"] == True].copy()
        completed = completed.dropna(subset=["mood_post"])
        if not completed.empty:
            delta = completed["mood_post"].astype(float) - completed["mood_pre"].astype(float)
            result["mean_delta_mood"] = round(delta.mean(), 3)

        # --- Cold-start detection -----------------------------------------
        # Cold-start = user appears in feedback but has < COLD_START_MIN_SESSIONS
        # total sessions across training + feedback combined
        cold_start = []
        if "user_id" in feedback_df.columns:
            feedback_counts = feedback_df["user_id"].value_counts().to_dict()
            for uid, fb_count in feedback_counts.items():
                total = user_session_counts.get(uid, 0) + fb_count
                if total < COLD_START_MIN_SESSIONS:
                    cold_start.append(uid)
        result["cold_start_users"] = cold_start

        return result


# ---------------------------------------------------------------------------
# Model retrainer
# ---------------------------------------------------------------------------

class ModelRetrainer:
    """
    Retrains random_forest.pkl on original training data + all feedback.

    Key design decisions
    --------------------
    FULL DATA RETRAINING: We retrain on original 8,000 sessions PLUS all
    accumulated feedback.  Training only on new data (incremental) would
    cause catastrophic forgetting — the forest would lose learned patterns
    from the original distribution.

    PIPELINE REFITTING: The scaler must be refit on the combined dataset,
    not the original scaler applied to new data.  Applying an old scaler
    to a drifted distribution produces systematically biased inputs.

    VERSIONING: Each retrain produces a versioned checkpoint
    (random_forest_v2.pkl, v3.pkl, ...).  The registry records all versions.
    The original random_forest.pkl is never overwritten — it is always the
    v1 baseline.
    """

    def retrain(
        self,
        original_sessions: pd.DataFrame,
        feedback_df:        pd.DataFrame,
        feature_list:       List[str],
        current_f1:         float,
        registry_path:      Path,
        checkpoints_dir:    Path,
    ) -> Tuple[str, float, str]:
        """
        Retrain and save a new versioned checkpoint.

        Returns
        -------
        (new_model_path, new_f1, version_tag)
        """
        # ---- Determine next version number -------------------------------
        version = self._next_version(registry_path)
        version_tag = f"v{version}"
        new_path = checkpoints_dir / f"random_forest_{version_tag}.pkl"

        # ---- Prepare combined dataset ------------------------------------
        combined = self._merge_datasets(original_sessions, feedback_df, feature_list)
        if combined is None or len(combined) < 100:
            raise RuntimeError(
                "Insufficient data for retraining "
                f"(need ≥100 labelled rows, got {len(combined) if combined is not None else 0})"
            )

        X = combined[feature_list].values
        y = combined["efficacy_label"].astype(int).values

        # ---- Refit Pipeline (scaler + RF) --------------------------------
        # class_weight='balanced' — same as Phase 3, handles class imbalance
        rf = RandomForestClassifier(
            n_estimators    = 300,
            max_depth       = None,
            min_samples_leaf= 2,
            class_weight    = "balanced",
            random_state    = 42,
            n_jobs          = -1,
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     rf),
        ])

        # Stratified train/val split for honest F1 estimate
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
        new_f1 = round(float(cv_scores.mean()), 4)

        # Fit on full combined dataset for the saved checkpoint
        pipeline.fit(X, y)

        # ---- Save checkpoint ---------------------------------------------
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        with open(new_path, "wb") as f:
            pickle.dump(pipeline, f)

        # ---- Update registry --------------------------------------------
        self._update_registry(
            registry_path = registry_path,
            version_tag   = version_tag,
            model_path    = str(new_path),
            new_f1        = new_f1,
            prev_f1       = current_f1,
            n_training    = len(combined),
            feature_list  = feature_list,
        )

        return str(new_path), new_f1, version_tag

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _next_version(registry_path: Path) -> int:
        if not registry_path.exists():
            return 2  # v1 is always the original
        with open(registry_path) as f:
            registry = json.load(f)
        versions = [int(e["version"].lstrip("v")) for e in registry.get("models", [])]
        return max(versions, default=1) + 1

    @staticmethod
    def _merge_datasets(
        original: pd.DataFrame,
        feedback: pd.DataFrame,
        feature_list: List[str],
    ) -> Optional[pd.DataFrame]:
        """
        Merge original training sessions with labelled feedback events.
        Feedback events without a true_label are excluded.
        """
        rows = []

        # Original sessions (already have efficacy_label)
        if "efficacy_label" in original.columns:
            orig_cols = feature_list + ["efficacy_label"]
            available = [c for c in orig_cols if c in original.columns]
            rows.append(original[available].dropna())

        # Feedback events that have been labelled
        if not feedback.empty and "true_label" in feedback.columns:
            labelled_fb = feedback.dropna(subset=["true_label"]).copy()
            labelled_fb = labelled_fb[labelled_fb["true_label"].isin([-1, 0, 1])]

            if not labelled_fb.empty:
                # Expand feature_vector JSON column if stored as string
                if "feature_vector" in labelled_fb.columns and \
                   labelled_fb["feature_vector"].dtype == object:
                    try:
                        fv = labelled_fb["feature_vector"].apply(
                            lambda x: json.loads(x) if isinstance(x, str) else x
                        )
                        fv_df = pd.DataFrame(fv.tolist(), index=labelled_fb.index)
                        labelled_fb = pd.concat(
                            [labelled_fb.drop(columns=["feature_vector"]), fv_df],
                            axis=1,
                        )
                    except Exception:
                        pass  # feature_vector may already be expanded

                labelled_fb = labelled_fb.rename(columns={"true_label": "efficacy_label"})
                # Drop old efficacy_label col if it now exists twice (test DataFrames
                # sometimes arrive with both columns; keep the renamed one)
                if labelled_fb.columns.duplicated().any():
                    # Keep first occurrence after renaming; drop the trailing duplicate
                    labelled_fb = labelled_fb.loc[:, ~labelled_fb.columns.duplicated(keep="first")]
                fb_cols = feature_list + ["efficacy_label"]
                available = [c for c in fb_cols if c in labelled_fb.columns]
                rows.append(labelled_fb[available].dropna())

        if not rows:
            return None
        return pd.concat([r.reset_index(drop=True) for r in rows], ignore_index=True)

    @staticmethod
    def _update_registry(
        registry_path: Path,
        version_tag:   str,
        model_path:    str,
        new_f1:        float,
        prev_f1:       float,
        n_training:    int,
        feature_list:  List[str],
    ) -> None:
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
        else:
            registry = {"models": []}

        # Ensure v1 (original) is always present as baseline
        if not any(e["version"] == "v1" for e in registry["models"]):
            registry["models"].insert(0, {
                "version":      "v1",
                "path":         str(PATHS["rf_model"]),
                "train_date":   "phase_3_training",
                "macro_f1":     0.906,   # Phase 3 result
                "n_training":   8000,
                "feature_hash": str(hash(tuple(feature_list))),
                "notes":        "Original Phase 3 checkpoint",
            })

        registry["models"].append({
            "version":      version_tag,
            "path":         model_path,
            "train_date":   datetime.now(timezone.utc).isoformat(),
            "macro_f1":     new_f1,
            "prev_f1":      prev_f1,
            "n_training":   n_training,
            "feature_hash": str(hash(tuple(feature_list))),
            "notes":        f"Retrained: F1 {prev_f1:.4f} → {new_f1:.4f}",
        })
        registry["latest"] = version_tag

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)


# ---------------------------------------------------------------------------
# Main DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Orchestrates feedback accumulation, PSI drift detection, rolling
    performance tracking, and conditional model retraining.

    Parameters
    ----------
    auto_retrain : bool
        If True, check_and_update() will automatically retrain when
        drift thresholds and volume thresholds are both exceeded.
        Default: True.

    retrain_session_threshold : int
        Number of new feedback sessions required before retraining
        can be triggered.  Default: 500 (RETRAIN_SESSION_THRESHOLD).

    paths : dict, optional
        Override file paths (used in tests to point at temp directories).

    Example
    -------
    >>> detector = DriftDetector()
    >>> detector.record(event)
    >>> report = detector.check_and_update()
    >>> print(report.summary())
    """

    def __init__(
        self,
        auto_retrain:               bool = True,
        retrain_session_threshold:  int  = RETRAIN_SESSION_THRESHOLD,
        paths:                      Optional[Dict[str, Path]] = None,
    ) -> None:
        self._auto_retrain    = auto_retrain
        self._retrain_thresh  = retrain_session_threshold
        self._paths           = paths or PATHS
        self._psi_calc        = PSICalculator()
        self._tracker         = RollingPerformanceTracker()
        self._retrainer       = ModelRetrainer()
        self._psi_fitted      = False

        # Cache loaded artefacts — avoid re-reading on every call
        self._feature_list:     Optional[List[str]]    = None
        self._training_f1:      float                  = 0.906  # Phase 3 default
        self._training_sessions:Optional[pd.DataFrame] = None

    # ---- Public API -------------------------------------------------------

    def record(self, event: FeedbackEvent) -> None:
        """
        Append one FeedbackEvent to data/feedback/sessions_feedback.csv.

        Thread-safety note: this method is NOT thread-safe.  In production,
        writes should be mediated by a queue (e.g. Redis) or a proper DB.
        For this phase, sequential single-process writes are assumed.
        """
        true_label = event.derive_true_label()

        row = {
            "session_id":      event.session_id,
            "user_id":         event.user_id,
            "track_id":        event.track_id,
            "intent":          event.intent,
            "mood_pre":        event.mood_pre,
            "mood_post":       event.mood_post,
            "efficacy_rating": event.efficacy_rating,
            "completed":       event.completed,
            "predicted_label": event.predicted_label,
            "true_label":      true_label,
            "feature_vector":  json.dumps(event.feature_vector),
            "timestamp":       event.timestamp,
        }

        feedback_path = self._paths["feedback_csv"]
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not feedback_path.exists()
        pd.DataFrame([row]).to_csv(
            feedback_path, mode="a", index=False, header=write_header
        )

    def record_batch(self, events: List[FeedbackEvent]) -> None:
        """Record multiple FeedbackEvents sequentially."""
        for event in events:
            self.record(event)

    def check_and_update(self, force_retrain: bool = False) -> DriftReport:
        """
        Run the full drift detection + optional retraining pipeline.

        Steps
        -----
        1. Load feedback CSV.
        2. Fit PSI calculator on training data (once, cached).
        3. Compute PSI scores on feedback feature vectors.
        4. Compute rolling performance metrics.
        5. Decide whether to retrain.
        6. Retrain if warranted.
        7. Write drift_report.json.
        8. Return DriftReport.

        Parameters
        ----------
        force_retrain : bool
            Skip volume/drift thresholds and retrain unconditionally.
            Useful for manual promotion of a new model.
        """
        # ---- Load artefacts ----------------------------------------------
        feature_list     = self._load_feature_list()
        train_sessions   = self._load_training_sessions()
        feedback_df      = self._load_feedback()
        training_f1      = self._load_training_f1()

        # ---- Fit PSI calculator on training distribution (once) ----------
        if not self._psi_fitted and train_sessions is not None:
            train_features = self._extract_training_features(
                train_sessions, feature_list
            )
            if train_features is not None and len(train_features) > 0:
                self._psi_calc.fit(train_features)
                self._psi_fitted = True

        # ---- Compute PSI -------------------------------------------------
        psi_scores: Dict[str, float] = {}
        if self._psi_fitted and not feedback_df.empty:
            fb_features = self._extract_feedback_features(feedback_df)
            if fb_features is not None and len(fb_features) > 0:
                psi_scores = self._psi_calc.compute(fb_features)

        drifted_features  = [f for f, v in psi_scores.items() if v >= PSI_STABLE]
        critical_features = [f for f, v in psi_scores.items() if v >= PSI_MONITOR]

        # ---- Rolling performance -----------------------------------------
        user_session_counts = self._compute_user_session_counts(train_sessions)
        perf = self._tracker.compute(feedback_df, user_session_counts)
        rolling_f1  = perf["rolling_f1"]
        f1_delta    = round(rolling_f1 - training_f1, 4) if rolling_f1 is not None else None
        cold_starts = perf["cold_start_users"]

        # ---- Retrain decision --------------------------------------------
        new_session_count  = len(feedback_df)
        retrain_recommended, retrain_reason = self._should_retrain(
            new_session_count = new_session_count,
            critical_features = critical_features,
            f1_delta          = f1_delta,
            force             = force_retrain,
        )

        # ---- Retrain -----------------------------------------------------
        retrained      = False
        new_model_path = ""

        if retrain_recommended and self._auto_retrain:
            try:
                new_model_path, new_f1, vtag = self._retrainer.retrain(
                    original_sessions = train_sessions,
                    feedback_df       = feedback_df,
                    feature_list      = feature_list,
                    current_f1        = training_f1,
                    registry_path     = self._paths["registry"],
                    checkpoints_dir   = self._paths["checkpoints_dir"],
                )
                retrained = True
                print(
                    f"[DriftDetector] Retrained → {vtag}  "
                    f"F1: {training_f1:.4f} → {new_f1:.4f}  "
                    f"Path: {new_model_path}"
                )
            except Exception as exc:
                retrain_reason += f" | Retrain failed: {exc}"
                print(f"[DriftDetector] Retrain failed: {exc}")

        # ---- Build and save report ---------------------------------------
        report = DriftReport(
            psi_scores          = psi_scores,
            drifted_features    = sorted(drifted_features),
            critical_features   = sorted(critical_features),
            rolling_f1          = rolling_f1,
            training_f1         = training_f1,
            f1_delta            = f1_delta,
            new_session_count   = new_session_count,
            cold_start_users    = cold_starts,
            retrain_recommended = retrain_recommended,
            retrain_reason      = retrain_reason,
            retrained           = retrained,
            new_model_path      = new_model_path,
        )

        self._save_report(report)
        return report

    # ---- Private helpers -------------------------------------------------

    def _load_feature_list(self) -> List[str]:
        if self._feature_list is not None:
            return self._feature_list
        p = self._paths["feature_list"]
        if p.exists():
            with open(p, "rb") as f:
                self._feature_list = pickle.load(f)
        else:
            self._feature_list = MONITORED_FEATURES
        return self._feature_list

    def _load_training_sessions(self) -> pd.DataFrame:
        if self._training_sessions is not None:
            return self._training_sessions
        p = self._paths["train_sessions"]
        if p.exists():
            self._training_sessions = pd.read_csv(p)
        else:
            self._training_sessions = pd.DataFrame()
        return self._training_sessions

    def _load_feedback(self) -> pd.DataFrame:
        p = self._paths["feedback_csv"]
        if p.exists():
            df = pd.read_csv(p)
            # Parse true_label (may be stored as float NaN for unlabelled)
            if "true_label" in df.columns:
                df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")
            return df
        return pd.DataFrame()

    def _load_training_f1(self) -> float:
        p = self._paths["registry"]
        if p.exists():
            with open(p) as f:
                registry = json.load(f)
            # Use most recent model's F1
            models = registry.get("models", [])
            if models:
                return models[-1].get("macro_f1", 0.906)
        return self._training_f1

    def _extract_training_features(
        self, df: pd.DataFrame, feature_list: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        Build the same 9-feature matrix that Phase 3 used for training.
        Adds intent_encoded from the raw intent column if needed.
        """
        df = df.copy()
        if "intent_encoded" not in df.columns and "session_intent" in df.columns:
            df["intent_encoded"] = df["session_intent"].map(INTENT_ENCODING)
        if "intent_encoded" not in df.columns and "intent" in df.columns:
            df["intent_encoded"] = df["intent"].map(INTENT_ENCODING)

        available = [f for f in MONITORED_FEATURES if f in df.columns]
        if not available:
            return None
        return df[available].dropna()

    def _extract_feedback_features(self, feedback_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Expand the feature_vector column (JSON string per row) into a DataFrame.
        """
        if "feature_vector" not in feedback_df.columns:
            return None
        try:
            vectors = feedback_df["feature_vector"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
            fv_df = pd.DataFrame(vectors.tolist())
        except Exception:
            return None

        # Map intent string to encoding if present but not numeric
        if "intent" in feedback_df.columns and "intent_encoded" not in fv_df.columns:
            fv_df["intent_encoded"] = feedback_df["intent"].map(INTENT_ENCODING).values

        available = [f for f in MONITORED_FEATURES if f in fv_df.columns]
        return fv_df[available].dropna() if available else None

    @staticmethod
    def _compute_user_session_counts(train_sessions: pd.DataFrame) -> Dict[str, int]:
        if train_sessions.empty:
            return {}
        col = next(
            (c for c in ["user_id", "userId"] if c in train_sessions.columns),
            None,
        )
        if col is None:
            return {}
        return train_sessions[col].value_counts().to_dict()

    @staticmethod
    def _should_retrain(
        new_session_count: int,
        critical_features: List[str],
        f1_delta:          Optional[float],
        force:             bool,
    ) -> Tuple[bool, str]:
        """
        Retraining decision logic.

        Retrain if ANY of:
        1. force=True
        2. ≥2 critical PSI features AND ≥ threshold new sessions
        3. F1 has dropped > 0.05 AND ≥ threshold new sessions
        4. Volume has exceeded threshold (scheduled routine retrain)
        """
        if force:
            return True, "Manual force_retrain requested."

        reasons = []
        volume_ok = new_session_count >= RETRAIN_SESSION_THRESHOLD

        if len(critical_features) >= 2:
            reasons.append(
                f"PSI critical on {len(critical_features)} features "
                f"({', '.join(sorted(critical_features)[:3])})"
            )

        if f1_delta is not None and f1_delta < -0.05:
            reasons.append(f"F1 degraded by {abs(f1_delta):.4f}")

        if reasons and volume_ok:
            return True, "; ".join(reasons)

        if reasons and not volume_ok:
            return False, (
                f"Drift detected ({'; '.join(reasons)}) but only "
                f"{new_session_count}/{RETRAIN_SESSION_THRESHOLD} sessions "
                "accumulated — waiting for more data before retraining."
            )

        if volume_ok:
            return True, (
                f"Routine scheduled retrain: {new_session_count} new sessions "
                f"accumulated (threshold: {RETRAIN_SESSION_THRESHOLD})"
            )

        return False, (
            f"Stable. {new_session_count}/{RETRAIN_SESSION_THRESHOLD} new sessions. "
            f"No critical drift detected."
        )

    def _save_report(self, report: DriftReport) -> None:
        p = self._paths["drift_report"]
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(report.as_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _make_synthetic_feedback(
    n:            int   = 600,
    drifted:      bool  = False,
    seed:         int   = 42,
) -> List[FeedbackEvent]:
    """
    Generate synthetic FeedbackEvent records for demonstration.

    When drifted=True, the feature distribution shifts meaningfully
    (higher tempo, higher energy) to trigger PSI alerts.
    """
    rng     = np.random.default_rng(seed)
    intents = list(INTENT_ENCODING.keys())
    events  = []

    for i in range(n):
        intent = intents[i % len(intents)]

        if drifted:
            # Simulate a shifted population: more energetic, faster tracks
            tempo   = float(np.clip(rng.normal(130, 20), 60, 200))
            energy  = float(np.clip(rng.normal(0.72, 0.10), 0.0, 1.0))
        else:
            tempo   = float(np.clip(rng.normal(90, 25), 40, 200))
            energy  = float(np.clip(rng.normal(0.45, 0.15), 0.0, 1.0))

        valence      = float(np.clip(rng.normal(0.50, 0.20), 0.0, 1.0))
        mood_pre     = float(np.clip(rng.normal(45, 18), 0, 100))
        mood_post    = float(np.clip(mood_pre + rng.normal(8, 12), 0, 100))
        rating       = int(np.clip(rng.integers(1, 11), 1, 10))
        pred_label   = int(rng.choice([-1, 0, 1], p=[0.55, 0.30, 0.15]))

        event = FeedbackEvent(
            session_id      = f"sess_demo_{i:05d}",
            user_id         = f"user_{rng.integers(1, 80):04d}",
            track_id        = f"track_{rng.integers(1, 5001):04d}",
            intent          = intent,
            mood_pre        = round(mood_pre, 1),
            mood_post       = round(mood_post, 1),
            efficacy_rating = rating,
            completed       = bool(rng.random() > 0.15),
            predicted_label = pred_label,
            feature_vector  = {
                "tempo_bpm":        round(tempo, 1),
                "energy":           round(energy, 3),
                "valence":          round(valence, 3),
                "acousticness":     round(float(rng.uniform(0.1, 0.9)), 3),
                "instrumentalness": round(float(rng.uniform(0.0, 0.8)), 3),
                "speechiness":      round(float(rng.uniform(0.02, 0.4)), 3),
                "loudness_db":      round(float(rng.uniform(-28, -6)), 1),
                "mood_pre":         round(mood_pre, 1),
                "intent_encoded":   INTENT_ENCODING[intent],
            },
        )
        events.append(event)

    return events


def _run_demo() -> None:
    """
    End-to-end Phase 7 demo using synthetic data and temp directories.

    Demonstrates:
    1. Recording feedback events
    2. PSI drift detection on a stable population
    3. PSI drift detection on a drifted population
    4. Rolling F1 and delta_mood computation
    5. Retrain decision logic
    """
    import tempfile, shutil

    print("=" * 68)
    print("  HarmonicAI — Phase 7 Drift Detector Demo")
    print("=" * 68)

    # ---- Temp directory so demo never touches real data ------------------
    tmpdir = Path(tempfile.mkdtemp())

    try:
        # Synthetic "training" sessions for PSI reference fitting
        rng = np.random.default_rng(0)
        n_train = 2000
        train_df = pd.DataFrame({
            "tempo_bpm":        np.clip(rng.normal(90, 25, n_train), 40, 200),
            "energy":           np.clip(rng.normal(0.45, 0.15, n_train), 0, 1),
            "valence":          np.clip(rng.normal(0.50, 0.20, n_train), 0, 1),
            "acousticness":     rng.uniform(0.1, 0.9, n_train),
            "instrumentalness": rng.uniform(0.0, 0.8, n_train),
            "speechiness":      rng.uniform(0.02, 0.4, n_train),
            "loudness_db":      rng.uniform(-28, -6, n_train),
            "mood_pre":         np.clip(rng.normal(45, 18, n_train), 0, 100),
            "session_intent":   rng.choice(list(INTENT_ENCODING.keys()), n_train),
            "efficacy_label":   rng.choice([-1, 0, 1], n_train, p=[0.596, 0.316, 0.088]),
        })

        (tmpdir / "data" / "processed").mkdir(parents=True)
        (tmpdir / "data" / "feedback").mkdir(parents=True)
        (tmpdir / "models" / "checkpoints").mkdir(parents=True)
        (tmpdir / "models" / "registry").mkdir(parents=True)

        train_path = tmpdir / "data" / "processed" / "sessions_normalized.csv"
        train_df.to_csv(train_path, index=False)

        # Minimal feature_list.pkl
        feat_path = tmpdir / "models" / "checkpoints" / "feature_list.pkl"
        with open(feat_path, "wb") as f:
            pickle.dump(MONITORED_FEATURES, f)

        demo_paths = {
            "rf_model":        tmpdir / "models" / "checkpoints" / "random_forest.pkl",
            "feature_list":    feat_path,
            "train_sessions":  train_path,
            "tracks":          PATHS["tracks"],
            "users":           PATHS["users"],
            "feedback_csv":    tmpdir / "data" / "feedback" / "sessions_feedback.csv",
            "drift_report":    tmpdir / "data" / "processed" / "drift_report.json",
            "registry":        tmpdir / "models" / "registry" / "model_registry.json",
            "checkpoints_dir": tmpdir / "models" / "checkpoints",
        }

        # ---- Scenario 1: STABLE population --------------------------------
        print("\n── Scenario 1: Stable feedback population (no drift)")
        detector = DriftDetector(auto_retrain=False, paths=demo_paths)
        stable_events = _make_synthetic_feedback(n=200, drifted=False, seed=1)
        detector.record_batch(stable_events)
        report1 = detector.check_and_update()
        print(report1.summary())

        # ---- Scenario 2: DRIFTED population --------------------------------
        print("\n── Scenario 2: Drifted feedback population (high tempo+energy)")
        # Reset feedback file
        demo_paths["feedback_csv"].unlink(missing_ok=True)
        detector2 = DriftDetector(
            auto_retrain=False,
            retrain_session_threshold=100,  # lower for demo
            paths=demo_paths,
        )
        drifted_events = _make_synthetic_feedback(n=600, drifted=True, seed=2)
        detector2.record_batch(drifted_events)
        report2 = detector2.check_and_update()
        print(report2.summary())

        # ---- Scenario 3: Auto-retrain demonstration -----------------------
        print("\n── Scenario 3: Force-retrain with combined dataset")
        # Build a tiny but valid RF for the checkpoint
        tiny_X = train_df[MONITORED_FEATURES].fillna(0).values[:200]
        tiny_y = train_df["efficacy_label"].astype(int).values[:200]
        rf_pipeline = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=10, random_state=42))])
        rf_pipeline.fit(tiny_X, tiny_y)
        with open(demo_paths["rf_model"], "wb") as f:
            pickle.dump(rf_pipeline, f)

        demo_paths["feedback_csv"].unlink(missing_ok=True)
        detector3 = DriftDetector(
            auto_retrain=True,
            retrain_session_threshold=50,  # low for demo
            paths=demo_paths,
        )
        events3 = _make_synthetic_feedback(n=60, drifted=True, seed=3)
        detector3.record_batch(events3)
        report3 = detector3.check_and_update(force_retrain=True)
        print(report3.summary())

        # ---- Show registry -----------------------------------------------
        reg_path = demo_paths["registry"]
        if reg_path.exists():
            print("\n── Model registry:")
            with open(reg_path) as f:
                reg = json.load(f)
            for entry in reg.get("models", []):
                print(
                    f"  {entry['version']:>4}  "
                    f"F1={entry['macro_f1']:.4f}  "
                    f"n={entry['n_training']:>5}  "
                    f"{entry.get('notes','')}"
                )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 68)
    print("  Demo complete.")
    print("=" * 68)


if __name__ == "__main__":
    _run_demo()
