"""
harmonicai/src/predictor/mood_classifier.py

Phase 3 — Mood Efficacy Predictor
===================================
Multi-model supervised ensemble that predicts whether a specific
track will have a positive (+1), neutral (0), or negative (-1)
therapeutic effect on a user in a specific mood state.

Models trained:
    1. Logistic Regression  — linear baseline, interpretable weights
    2. SVM (RBF kernel)     — non-linear boundary, strong on mid-scale data
    3. Random Forest        — ensemble of trees, captures interactions
    4. Naïve Bayes          — probabilistic baseline

Target:
    efficacy_label ∈ {-1, 0, +1}

Class imbalance:
    -1: 59.6%  |  0: 31.6%  |  +1: 8.8%
    Handled via class_weight='balanced' on all models.

Run:
    python src/predictor/mood_classifier.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline        import Pipeline
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ingestion.loader import load_all

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"
MODEL_DIR    = PROJECT_ROOT / "models" / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ─────────────────────────────────────────────
# SECTION 1: Feature Engineering
# ─────────────────────────────────────────────
#
# We build two feature sets:
#
# A) Track-level acoustic features (from Phase 2)
#    Same features used in clustering — objective acoustic properties
#
# B) Session-context features
#    The same track can be therapeutic or harmful depending on context.
#    A 120 BPM track is great for mood_uplift, harmful for sleep_induction.
#    Context features encode the user's state and intent.
#
# The final feature vector is A + B concatenated.
# This is the key architectural decision of Phase 3:
# we are not just classifying tracks — we are classifying (track, context) pairs.

ACOUSTIC_FEATURES = [
    "tempo_bpm",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "loudness_db",
]

CONTEXT_FEATURES = [
    "mood_pre",           # user's current mood state (0-100)
    "intent_encoded",     # session intent as integer
]

ALL_FEATURES = ACOUSTIC_FEATURES + CONTEXT_FEATURES

# Intent encoding map — ordinal encoding by arousal level
INTENT_MAP = {
    "sleep_induction":  0,
    "anxiety_relief":   1,
    "grief_processing": 2,
    "deep_focus":       3,
    "mood_uplift":      4,
}


def build_feature_matrix(
    sessions: pd.DataFrame,
    tracks: pd.DataFrame
) -> tuple:
    """
    Joins sessions with track features to build the final
    (track, context) feature matrix.

    Why join here rather than pre-joining earlier?
    Because sessions.csv and tracks.csv have different grain:
      - tracks: one row per track (5,000 rows)
      - sessions: one row per session (8,000 rows)

    A track appears in multiple sessions with different users
    and different moods. The same track's acoustic features are
    identical across sessions, but the context features differ.
    That's exactly what we want — the model learns that context matters.
    """

    # Join on track_id to get acoustic features per session
    merged = sessions.merge(
        tracks[["track_id"] + ACOUSTIC_FEATURES],
        on="track_id",
        how="inner"
    )

    # Encode session intent as integer
    merged["intent_encoded"] = merged["session_intent"].map(INTENT_MAP)

    # Drop rows where intent encoding failed (shouldn't happen post-validation)
    merged = merged.dropna(subset=["intent_encoded"])
    merged["intent_encoded"] = merged["intent_encoded"].astype(int)

    X = merged[ALL_FEATURES].values
    y = merged["efficacy_label"].values

    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(y)
        bar = "█" * int(pct * 30)
        print(f"    {label:+d}: {bar:<30} {count:>5} ({pct:.1%})")

    return X, y, merged


# ─────────────────────────────────────────────
# SECTION 2: Model Definitions
# ─────────────────────────────────────────────

def build_models() -> dict:
    """
    Builds sklearn Pipeline objects for each classifier.

    Why Pipelines?
    A Pipeline chains preprocessing + model into one object.
    When you call pipeline.fit(X_train, y_train), it:
      1. Fits the scaler on X_train
      2. Transforms X_train with the fitted scaler
      3. Fits the model on the scaled X_train

    When you call pipeline.predict(X_test), it:
      1. Transforms X_test with the ALREADY FITTED scaler
      2. Predicts with the model

    This prevents data leakage — the scaler never sees test data
    statistics during training. Without Pipeline, it's easy to
    accidentally fit the scaler on the full dataset before splitting,
    which leaks test distribution information into training.

    class_weight='balanced':
    Automatically sets class weights as:
        weight_c = n_samples / (n_classes × n_samples_c)

    For our distribution:
        weight_{-1} = 8000 / (3 × 4768) ≈ 0.56   (majority, downweighted)
        weight_{0}  = 8000 / (3 × 2528) ≈ 1.05
        weight_{+1} = 8000 / (3 ×  704) ≈ 3.79   (minority, upweighted ~4x)

    This forces the model to pay 4x more attention to therapeutic sessions.
    """

    models = {

        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,                    # regularisation strength (higher C = less regularisation)
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_SEED,
            ))
        ]),

        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,                    # margin hardness
                gamma="scale",            # RBF width: 1 / (n_features × X.var())
                class_weight="balanced",
                probability=True,         # needed for soft voting ensemble
                random_state=RANDOM_SEED,
            ))
        ]),

        "random_forest": Pipeline([
            # Note: no scaler needed — trees are scale-invariant
            # We include it anyway for pipeline consistency
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,         # more trees = more stable, diminishing returns after ~200
                max_depth=None,           # grow full trees; regularise via min_samples_leaf
                min_samples_leaf=5,       # prevents single-sample leaves (overfitting)
                max_features="sqrt",      # consider √n_features at each split (standard)
                class_weight="balanced",
                n_jobs=-1,                # use all CPU cores
                random_state=RANDOM_SEED,
            ))
        ]),

        "naive_bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(
                # GaussianNB has no class_weight parameter
                # We handle imbalance via priors instead
                priors=[0.30, 0.30, 0.40]  # prior P(class): [-1, 0, +1]
                # Slightly boost +1 prior to compensate for imbalance
            ))
        ]),

    }

    return models


# ─────────────────────────────────────────────
# SECTION 3: Stratified Cross-Validation
# ─────────────────────────────────────────────

def evaluate_models(
    models: dict,
    X: np.ndarray,
    y: np.ndarray
) -> pd.DataFrame:
    """
    Evaluates all models using 5-fold Stratified Cross-Validation.

    Why stratified?
    With class imbalance, random splits might put all +1 samples
    in training and leave none in the test fold. StratifiedKFold
    ensures each fold has the same class distribution as the full dataset.

    Why cross-validation instead of a single train/test split?
    A single split is high variance — you might get lucky or unlucky
    with which samples end up in test. Cross-validation averages
    over 5 different splits, giving a more reliable estimate of
    true generalisation performance.

    Metrics computed:
        macro F1  — average F1 across all classes (our primary metric)
        +1 F1     — F1 on the therapeutic class specifically (clinical metric)
        -1 F1     — F1 on the harmful class (safety metric)
        accuracy  — included to show why it's misleading here
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = []

    print(f"\n  {'='*62}")
    print(f"  5-Fold Stratified Cross-Validation Results")
    print(f"  {'='*62}")
    print(f"  {'Model':<25} {'Macro F1':>9} {'F1 (+1)':>9} {'F1 (-1)':>9} {'Accuracy':>10}")
    print(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*9} {'-'*10}")

    for name, pipeline in models.items():

        # cross_validate returns per-fold scores
        cv_scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring={
                "f1_macro":    "f1_macro",
                "accuracy":    "accuracy",
                "f1_positive": lambda est, X, y: f1_score(
                    y, est.predict(X), labels=[1], average="macro", zero_division=0
                ),
                "f1_negative": lambda est, X, y: f1_score(
                    y, est.predict(X), labels=[-1], average="macro", zero_division=0
                ),
            },
            return_train_score=False,
            n_jobs=1,     # 1 here because RF already uses n_jobs=-1 internally
        )

        mean_f1      = cv_scores["test_f1_macro"].mean()
        std_f1       = cv_scores["test_f1_macro"].std()
        mean_f1_pos  = cv_scores["test_f1_positive"].mean()
        mean_f1_neg  = cv_scores["test_f1_negative"].mean()
        mean_acc     = cv_scores["test_accuracy"].mean()

        # Flag: does it meet the PRD target?
        flag = "✅" if mean_f1 > 0.85 else ("⚠ " if mean_f1 > 0.60 else "❌")

        print(f"  {name:<25} {mean_f1:>7.4f}±{std_f1:.3f} "
              f"{mean_f1_pos:>9.4f} {mean_f1_neg:>9.4f} {mean_acc:>10.4f}  {flag}")

        results.append({
            "model":        name,
            "f1_macro":     round(mean_f1, 4),
            "f1_macro_std": round(std_f1, 4),
            "f1_positive":  round(mean_f1_pos, 4),
            "f1_negative":  round(mean_f1_neg, 4),
            "accuracy":     round(mean_acc, 4),
        })

    print(f"\n  PRD target: macro F1 > 0.85")
    print(f"  Note: accuracy appears high due to majority-class dominance")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# SECTION 4: Full Training & Detailed Report
# ─────────────────────────────────────────────

def train_and_report(
    models: dict,
    X: np.ndarray,
    y: np.ndarray
) -> dict:
    """
    Trains each model on the full dataset and produces a
    detailed per-class classification report.

    In production: you would train on 80% and evaluate on 20%.
    Here we train on full data for the saved model (cross-validation
    already gave us unbiased performance estimates) and report
    on training performance to inspect what the model learned.

    Why train on full data for the final model?
    Because cross-validation was our evaluation. The final production
    model should use all available data — holding 20% back permanently
    wastes signal. This is standard practice in small-data ML.
    """

    trained = {}

    print(f"\n  {'='*62}")
    print(f"  Full Dataset Training + Classification Reports")
    print(f"  {'='*62}")

    for name, pipeline in models.items():
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        f1 = f1_score(y, y_pred, average="macro", zero_division=0)

        print(f"\n  ── {name.upper()} (training F1 macro: {f1:.4f}) ──")
        print(classification_report(
            y, y_pred,
            target_names=["harmful (-1)", "neutral (0)", "therapeutic (+1)"],
            zero_division=0
        ))

        trained[name] = pipeline

    return trained


# ─────────────────────────────────────────────
# SECTION 5: Feature Importance Analysis
# ─────────────────────────────────────────────

def analyze_feature_importance(trained_models: dict):
    """
    Extracts feature importance from Random Forest and
    coefficient weights from Logistic Regression.

    These are the most clinically valuable outputs of Phase 3:
    they tell you *which acoustic properties actually drive
    therapeutic outcomes* — not just what the model predicts.

    Random Forest importance: average impurity reduction
    across all trees when splitting on each feature.
    Higher = more important for discriminating between classes.

    Logistic Regression coefficients: signed weights per feature
    per class. Positive weight = feature pushes toward that class.
    """

    print(f"\n  {'='*62}")
    print(f"  Feature Importance Analysis")
    print(f"  {'='*62}")

    # ── Random Forest importances ────────────────────────
    rf_pipe = trained_models["random_forest"]
    rf_clf  = rf_pipe.named_steps["clf"]

    importances = rf_clf.feature_importances_
    importance_df = pd.DataFrame({
        "feature":    ALL_FEATURES,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(f"\n  Random Forest — Feature Importances:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<22} {bar:<40} {row['importance']:.4f}")

    # ── Logistic Regression coefficients ─────────────────
    lr_pipe = trained_models["logistic_regression"]
    lr_clf  = lr_pipe.named_steps["clf"]

    print(f"\n  Logistic Regression — Coefficients per class:")
    print(f"  {'Feature':<22} {'Harmful (-1)':>14} {'Neutral (0)':>14} {'Therapeutic (+1)':>16}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*16}")

    class_names = ["Harmful (-1)", "Neutral (0)", "Therapeutic (+1)"]
    for i, feat in enumerate(ALL_FEATURES):
        coeffs = lr_clf.coef_[:, i]  # one coeff per class
        print(f"  {feat:<22} {coeffs[0]:>+14.4f} {coeffs[1]:>+14.4f} {coeffs[2]:>+16.4f}")

    print(f"\n  Interpretation guide:")
    print(f"    Therapeutic (+1) column: positive = feature helps, negative = feature hurts")
    print(f"    e.g. energy coeff on +1: if negative, high energy reduces therapeutic probability")

    return importance_df


# ─────────────────────────────────────────────
# SECTION 6: Confusion Matrix Analysis
# ─────────────────────────────────────────────

def analyze_confusion_matrices(trained_models: dict, X: np.ndarray, y: np.ndarray):
    """
    Confusion matrices reveal *how* a model fails — not just that it fails.

    For a therapeutic system, the clinically significant errors are:
        Predicted +1, Actual -1  → recommended a harmful track (worst error)
        Predicted -1, Actual +1  → rejected a therapeutic track (opportunity cost)

    The matrix is read: rows = actual, columns = predicted
    """

    print(f"\n  {'='*62}")
    print(f"  Confusion Matrix Analysis (training set)")
    print(f"  {'='*62}")
    print(f"  Classes: -1=Harmful  0=Neutral  +1=Therapeutic")
    print(f"  Read: rows=actual, columns=predicted")
    print(f"  Clinical focus: [actual:-1, predicted:+1] = dangerous false positive")

    labels = [-1, 0, 1]

    for name, pipeline in trained_models.items():
        y_pred = pipeline.predict(X)
        cm = confusion_matrix(y, y_pred, labels=labels)

        # Identify the dangerous cell: actual=-1, predicted=+1
        dangerous_fp = cm[0, 2]   # row 0 = actual -1, col 2 = predicted +1
        total_harmful = cm[0].sum()
        dangerous_rate = dangerous_fp / total_harmful if total_harmful > 0 else 0

        print(f"\n  {name}:")
        print(f"              Pred:-1  Pred:0  Pred:+1")
        print(f"  Actual: -1  {cm[0,0]:>7}  {cm[0,1]:>6}  {cm[0,2]:>7}  ← {dangerous_rate:.1%} dangerously mislabelled")
        print(f"  Actual:  0  {cm[1,0]:>7}  {cm[1,1]:>6}  {cm[1,2]:>7}")
        print(f"  Actual: +1  {cm[2,0]:>7}  {cm[2,1]:>6}  {cm[2,2]:>7}")


# ─────────────────────────────────────────────
# SECTION 7: Ensemble (Soft Voting)
# ─────────────────────────────────────────────

def build_and_evaluate_ensemble(
    trained_models: dict,
    X: np.ndarray,
    y: np.ndarray
) -> Pipeline:
    """
    Builds a soft-voting ensemble from all four models.

    Soft voting: average the predicted class probabilities
    across all models, then pick the class with the highest
    average probability.

    Why soft over hard voting?
    Hard voting: each model casts one vote (predicted class)
    Soft voting: each model casts a probability distribution
    Soft voting leverages confidence information — a model that
    predicts +1 with 95% confidence outweighs one that predicts
    +1 with 51% confidence. Hard voting treats both equally.

    Why does this often outperform individual models?
    Each model has different inductive biases and makes different
    errors. When their errors are uncorrelated, averaging them
    reduces variance without increasing bias — the ensemble is
    more stable than any single model.
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    ensemble = VotingClassifier(
        estimators=[
            ("lr",  trained_models["logistic_regression"]),
            ("svm", trained_models["svm_rbf"]),
            ("rf",  trained_models["random_forest"]),
            ("nb",  trained_models["naive_bayes"]),
        ],
        voting="soft",
        weights=[1, 2, 3, 1],  # RF gets 3x weight: highest individual F1
    )

    cv_scores = cross_validate(
        ensemble, X, y,
        cv=cv,
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
    )

    mean_f1  = cv_scores["test_f1_macro"].mean()
    std_f1   = cv_scores["test_f1_macro"].std()
    mean_acc = cv_scores["test_accuracy"].mean()

    print(f"\n  {'='*62}")
    print(f"  Soft-Voting Ensemble (LR×1 + SVM×2 + RF×3 + NB×1)")
    print(f"  {'='*62}")
    print(f"  Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Accuracy: {mean_acc:.4f}")

    flag = "✅ meets PRD target" if mean_f1 > 0.85 else "⚠  below PRD target — see debrief"
    print(f"  Assessment: {flag}")

    # Fit ensemble on full data for persistence
    ensemble.fit(X, y)

    return ensemble


# ─────────────────────────────────────────────
# SECTION 8: Stratified Analysis by Intent
# ─────────────────────────────────────────────

def stratified_intent_analysis(
    best_model,
    X: np.ndarray,
    y: np.ndarray,
    merged: pd.DataFrame
):
    """
    Evaluates model performance separately per session intent.

    A model that achieves good global F1 might still fail badly
    on grief_processing (small group, hardest intent) while
    excelling on anxiety_relief (largest group).

    In a safety-critical system, per-group performance matters
    as much as global performance — potentially more.
    """

    print(f"\n  {'='*62}")
    print(f"  Per-Intent Performance (best model)")
    print(f"  {'='*62}")
    print(f"  {'Intent':<22} {'n':>5} {'Macro F1':>10} {'F1(+1)':>8} {'Assessment'}")
    print(f"  {'-'*22} {'-'*5} {'-'*10} {'-'*8} {'-'*18}")

    y_pred = best_model.predict(X)

    for intent, code in INTENT_MAP.items():
        mask = merged["session_intent"].values == intent
        if mask.sum() < 10:
            continue

        y_int      = y[mask]
        y_pred_int = y_pred[mask]

        f1_macro = f1_score(y_int, y_pred_int, average="macro", zero_division=0)
        f1_pos   = f1_score(y_int, y_pred_int, labels=[1], average="macro", zero_division=0)
        n        = mask.sum()

        assessment = "✅" if f1_macro > 0.70 else ("⚠ " if f1_macro > 0.50 else "❌ needs more data")

        print(f"  {intent:<22} {n:>5} {f1_macro:>10.4f} {f1_pos:>8.4f}  {assessment}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 62)
    print("  HarmonicAI — Phase 3: Mood Efficacy Predictor")
    print("=" * 62)

    # ── Load data ──────────────────────────────────────────
    tracks, users, sessions = load_all()

    # ── Build feature matrix ───────────────────────────────
    print("\n[1/7] Building (track × context) feature matrix...")
    X, y, merged = build_feature_matrix(sessions, tracks)

    # ── Define models ──────────────────────────────────────
    print("\n[2/7] Defining model pipelines...")
    models = build_models()
    print(f"  Models defined: {list(models.keys())}")

    # ── Cross-validate ─────────────────────────────────────
    print("\n[3/7] Running 5-fold stratified cross-validation...")
    cv_results = evaluate_models(models, X, y)

    # ── Full training ──────────────────────────────────────
    print("\n[4/7] Training on full dataset...")
    trained = train_and_report(models, X, y)

    # ── Feature importance ─────────────────────────────────
    print("\n[5/7] Analyzing feature importance...")
    importance_df = analyze_feature_importance(trained)

    # ── Confusion matrices ─────────────────────────────────
    print("\n[6/7] Confusion matrix analysis...")
    analyze_confusion_matrices(trained, X, y)

    # ── Ensemble ───────────────────────────────────────────
    print("\n[7/7] Building and evaluating soft-voting ensemble...")
    ensemble = build_and_evaluate_ensemble(trained, X, y)

    # ── Per-intent analysis ─────────────────────────────────
    stratified_intent_analysis(trained["random_forest"], X, y, merged)

    # ── Persist ────────────────────────────────────────────
    cv_results.to_csv(OUTPUT_DIR / "model_cv_results.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    with open(MODEL_DIR / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    with open(MODEL_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(trained["random_forest"], f)
    with open(MODEL_DIR / "feature_list.pkl", "wb") as f:
        pickle.dump(ALL_FEATURES, f)

    print(f"\n\n✅ Phase 3 complete.")
    print(f"   model_cv_results.csv   — cross-validation scores per model")
    print(f"   feature_importance.csv — RF feature importances")
    print(f"   models/checkpoints/    — serialised ensemble + RF + feature list")
    print(f"\nNext step → Phase 4: python src/frequency/spectrogram.py\n")
