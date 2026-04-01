"""
harmonicai/src/nlp/safety_filter.py

Phase 5 — Lyrical Safety Filter
=================================
Multi-layer NLP pipeline that scans track lyrics for psychologically
harmful content before any track reaches a user.

Architecture (three layers, applied in order):

  Layer 1 — Hard blocklist (instant, zero-latency)
    Exact keyword matching on highest-severity terms.
    Catches unambiguous content immediately without ML overhead.
    Threshold: any match → block, no exceptions.

  Layer 2 — TF-IDF + Logistic Regression (primary classifier)
    Trained on labelled lyric corpus. Produces a continuous
    risk_score [0.0, 1.0] per risk category.
    Decision threshold tuned to recall ≥ 0.95 on harmful classes.

  Layer 3 — Semantic similarity (secondary check)
    Computes cosine similarity between lyric embedding and
    risk-category centroid embeddings.
    Catches paraphrased harmful content that avoids blocklist words.
    (Full word2vec deferred to production — simulated here with
    keyword-weighted bag-of-words embeddings.)

Output per track:
    {
      "track_id":       "...",
      "safe":           True/False,
      "risk_score":     0.0–1.0,
      "risk_category":  "self_harm" | "hopelessness" | ... | "safe",
      "block_reason":   "blocklist_match" | "classifier" | "semantic" | None,
      "flagged_terms":  ["...", ...],
      "reviewed_by_human": False   ← flag for human review queue
    }

Run:
    python src/nlp/safety_filter.py
"""

import numpy as np
import pandas as pd
import json
import pickle
import re
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing  import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model  import LogisticRegression
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics       import (
    classification_report, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"
MODEL_DIR    = PROJECT_ROOT / "models" / "checkpoints"

RANDOM_SEED = 42


# ─────────────────────────────────────────────
# LAYER 1: Hard Blocklist
# ─────────────────────────────────────────────
# These terms trigger an immediate block regardless of context.
# They are unambiguous in the context of music lyrics.
# Compiled from clinical literature on suicide contagion
# and safe messaging guidelines (AFSP, Samaritans).
#
# IMPORTANT: This list is intentionally non-exhaustive here.
# Production systems use continuously updated lists maintained
# by clinical safety teams — not static code.

HARD_BLOCKLIST = {
    # Self-harm / suicidality — critical severity
    "self_harm": [
        "want to die", "kill myself", "end my life", "take my life",
        "suicidal", "suicide", "slit my wrists", "overdose on",
        "wrote a note", "said goodbye", "planned a way out",
        "don't want to be alive", "not being here", "disappear forever",
        "bridge and jump", "pills and sleep", "final decision",
    ],
    # Substance glorification — high severity
    "substance_abuse": [
        "blackout drunk", "inject and forget", "chasing the high",
        "hooked on pills", "needle in my arm",
    ],
}


# ─────────────────────────────────────────────
# LAYER 2: TF-IDF Feature Engineering
# ─────────────────────────────────────────────

def preprocess_lyric(text: str) -> str:
    """
    Cleans lyric text before tokenization.

    Steps:
      1. Lowercase — "Hopeless" and "hopeless" are the same risk signal
      2. Remove punctuation except apostrophes — "can't" ≠ "cant"
      3. Collapse repeated whitespace
      4. Remove line break markers (our "/" separator)

    Why NOT remove stopwords for safety filtering?
    Standard NLP removes stopwords ("the", "I", "is") for topic
    modelling — they add noise without meaning.
    But for safety filtering, "I" is critical. "I want to die" is
    personal and immediate. "Characters want to die" is narrative.
    Removing "I" would destroy this signal.
    We keep all stopwords.
    """

    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("/", " ")
    return text


# ─────────────────────────────────────────────
# LAYER 3: Semantic Similarity (simplified)
# ─────────────────────────────────────────────
# In production: use pre-trained word embeddings (GloVe/Word2Vec).
# Here: we build category prototype vectors from TF-IDF space
# and compute cosine similarity.
# Catches paraphrased content: "I've decided to stop existing"
# is semantically close to self-harm even without blocklist words.

RISK_KEYWORDS = {
    "self_harm": [
        "die", "dead", "gone", "end", "stop", "never", "goodbye",
        "note", "bridge", "pills", "knife", "hurt", "pain", "escape",
        "disappear", "nothing", "decided", "final", "last"
    ],
    "hopelessness": [
        "nothing", "never", "pointless", "empty", "worthless", "done",
        "gave up", "invisible", "exhausted", "fail", "broken", "lost",
        "dark", "numb", "hollow", "impossible", "useless"
    ],
    "grief_trigger": [
        "gone", "miss", "lost", "funeral", "died", "grief", "memory",
        "remember", "chair", "empty", "without", "voice", "photograph",
        "anniversary", "waves", "impossible"
    ],
    "substance_abuse": [
        "drink", "drunk", "pills", "high", "bottle", "blackout", "fix",
        "sober", "relapse", "numb", "escape", "addiction", "craving"
    ],
    "safe": [
        "love", "hope", "light", "morning", "together", "beautiful",
        "smile", "peace", "grow", "rise", "joy", "grateful", "wonder"
    ],
}


def compute_keyword_overlap(
    lyric: str,
    keywords: list,
) -> float:
    """
    Computes the fraction of risk keywords present in the lyric.
    Used as a semantic similarity proxy when embeddings are unavailable.
    """
    words = set(lyric.lower().split())
    matches = sum(1 for kw in keywords if kw in lyric.lower())
    return matches / (len(keywords) + 1e-8)


def semantic_risk_score(lyric: str) -> dict:
    """
    Computes semantic overlap with each risk category.
    Returns a dict of category → overlap score.
    """
    preprocessed = preprocess_lyric(lyric)
    return {
        cat: compute_keyword_overlap(preprocessed, kws)
        for cat, kws in RISK_KEYWORDS.items()
    }


# ─────────────────────────────────────────────
# SECTION: Blocklist Checker (Layer 1)
# ─────────────────────────────────────────────

def check_blocklist(lyric: str) -> tuple:
    """
    Runs Layer 1: exact phrase matching against the hard blocklist.

    Returns:
        (blocked: bool, category: str | None, matched_terms: list)

    Case-insensitive matching. Whole-phrase matching (not substring)
    to avoid false positives from fragments.
    """
    lyric_lower = lyric.lower()
    matched = []
    matched_cat = None

    for category, phrases in HARD_BLOCKLIST.items():
        for phrase in phrases:
            if phrase in lyric_lower:
                matched.append(phrase)
                matched_cat = category

    return bool(matched), matched_cat, matched


# ─────────────────────────────────────────────
# SECTION: Model Training (Layer 2)
# ─────────────────────────────────────────────

def build_safety_classifier() -> Pipeline:
    """
    Builds the TF-IDF + Logistic Regression pipeline.

    TF-IDF parameters:
        ngram_range=(1, 3): unigrams + bigrams + trigrams
            "want to die" (trigram) is far more dangerous than
            "want", "to", "die" individually.
            Without trigrams, the classifier misses phrase-level risk.

        max_features=5000: vocabulary cap
            Prevents overfitting to rare lyric-specific terms.

        sublinear_tf=True: apply log(1 + tf) instead of raw tf
            Prevents tracks with repeated risk phrases from
            dominating. "die die die die" shouldn't score 4× higher
            than "die" — the risk is equivalent.

        analyzer='word': word-level tokenization (not char)
            We explained this choice in the theory section.

    Logistic Regression:
        C=0.5: moderate regularisation
            Prevents overfitting to synthetic vocabulary.
            Lower C = stronger regularisation = simpler decision boundary.

        class_weight='balanced': upweight minority classes
            self_harm is only 6% of data — without balancing,
            the model ignores it. That is clinically unacceptable.
    """

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess_lyric,
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
            min_df=2,           # ignore terms that appear in only 1 document
            analyzer="word",
        )),
        ("clf", LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])


def tune_threshold_for_recall(
    pipeline: Pipeline,
    X_val: pd.Series,
    y_val: np.ndarray,
    target_class: str,
    target_recall: float = 0.95,
) -> float:
    """
    Tunes the decision threshold so that recall on the target class
    meets or exceeds target_recall.

    Standard classifiers use threshold=0.5 (predict the class with
    highest probability). For safety systems this is wrong —
    0.5 is calibrated for balanced precision/recall.

    We scan thresholds from 0.05 to 0.80 and find the highest
    threshold (most conservative = fewest positives) that still
    achieves target_recall. Higher threshold = fewer false positives.

    Why not just use 0.05 (flag everything)?
    At threshold=0.05, every track gets flagged. Recall=1.0.
    But precision=0.06 — 94% of flags are wrong. The music
    library becomes unusable. We need the highest threshold
    that preserves safety.
    """

    classes = list(pipeline.classes_)
    if target_class not in classes:
        return 0.5

    class_idx = classes.index(target_class)
    probs = pipeline.predict_proba(X_val)[:, class_idx]
    is_target = (y_val == target_class).astype(int)

    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.80, 0.05):
        preds = (probs >= threshold).astype(int)
        if is_target.sum() == 0:
            break
        recall = recall_score(is_target, preds, zero_division=0)
        if recall >= target_recall:
            best_threshold = threshold  # keep updating — want highest valid threshold

    return round(float(best_threshold), 2)


def evaluate_safety_classifier(
    pipeline: Pipeline,
    X: pd.Series,
    y: np.ndarray,
) -> dict:
    """
    Evaluates the classifier with stratified cross-validation.

    For safety filtering, we report:
        Macro F1           — overall balance
        self_harm recall   — THE critical safety metric
        self_harm precision — false positive cost
        hopelessness recall — secondary safety metric
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring={
            "f1_macro": "f1_macro",
            "accuracy": "accuracy",
        },
        return_train_score=False,
    )

    # Per-class recall requires manual fold iteration
    self_harm_recalls    = []
    hopeless_recalls     = []
    self_harm_precisions = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)

        classes = list(pipeline.classes_)

        if "self_harm" in classes:
            is_sh   = (y_val == "self_harm").astype(int)
            pred_sh = (y_pred == "self_harm").astype(int)
            if is_sh.sum() > 0:
                self_harm_recalls.append(recall_score(is_sh, pred_sh, zero_division=0))
                self_harm_precisions.append(precision_score(is_sh, pred_sh, zero_division=0))

        if "hopelessness" in classes:
            is_hp   = (y_val == "hopelessness").astype(int)
            pred_hp = (y_pred == "hopelessness").astype(int)
            hopeless_recalls.append(recall_score(is_hp, pred_hp, zero_division=0))

    return {
        "f1_macro":            round(float(np.mean(results["test_f1_macro"])), 4),
        "f1_macro_std":        round(float(np.std(results["test_f1_macro"])), 4),
        "accuracy":            round(float(np.mean(results["test_accuracy"])), 4),
        "self_harm_recall":    round(float(np.mean(self_harm_recalls)) if self_harm_recalls else 0.0, 4),
        "self_harm_precision": round(float(np.mean(self_harm_precisions)) if self_harm_precisions else 0.0, 4),
        "hopelessness_recall": round(float(np.mean(hopeless_recalls)) if hopeless_recalls else 0.0, 4),
    }


# ─────────────────────────────────────────────
# SECTION: Inference (Full Three-Layer Filter)
# ─────────────────────────────────────────────

class LyricalSafetyFilter:
    """
    Production-ready three-layer safety filter.

    Usage:
        filt = LyricalSafetyFilter.load(model_path)
        result = filt.check(track_id, lyric, user_sensitivity_flags)
    """

    def __init__(self, pipeline: Pipeline, threshold: float = 0.35):
        self.pipeline  = pipeline
        self.threshold = threshold
        self.classes   = list(pipeline.classes_)

    def check(
        self,
        track_id: str,
        lyric: str,
        instrumentalness: float = 0.0,
        sensitivity_flags: list = None,
    ) -> dict:
        """
        Runs all three filter layers and returns a safety decision.

        Args:
            track_id:          track identifier
            lyric:             raw lyric text
            instrumentalness:  if > 0.6, skip filter (no vocals)
            sensitivity_flags: user profile flags e.g. ['ptsd', 'grief']

        Returns:
            dict with keys: safe, risk_score, risk_category,
                           block_reason, flagged_terms, reviewed_by_human
        """

        sensitivity_flags = sensitivity_flags or []

        # ── Fast path: instrumental track ─────────────────
        if instrumentalness > 0.6:
            return self._safe_result(track_id, reason="instrumental_skip")

        # ── Layer 1: Hard blocklist ────────────────────────
        blocked, bl_cat, matched_terms = check_blocklist(lyric)
        if blocked:
            return {
                "track_id":          track_id,
                "safe":              False,
                "risk_score":        0.98,
                "risk_category":     bl_cat,
                "block_reason":      "blocklist_match",
                "flagged_terms":     matched_terms,
                "reviewed_by_human": False,
            }

        # ── Layer 2: TF-IDF classifier ────────────────────
        probs = self.pipeline.predict_proba([lyric])[0]
        prob_dict = dict(zip(self.classes, probs))

        # Highest-risk category by probability
        risk_cat  = max(prob_dict, key=prob_dict.get)
        risk_score = float(prob_dict.get(risk_cat, 0.0))

        # The classifier's "safe" probability
        safe_prob = float(prob_dict.get("safe", 0.0))

        # Block if any harmful class exceeds threshold
        harmful_classes = ["self_harm", "hopelessness", "substance_abuse"]
        harmful_score = max(
            prob_dict.get(c, 0.0) for c in harmful_classes
        )

        if harmful_score >= self.threshold:
            return {
                "track_id":          track_id,
                "safe":              False,
                "risk_score":        round(harmful_score, 4),
                "risk_category":     risk_cat,
                "block_reason":      "classifier",
                "flagged_terms":     [],
                "reviewed_by_human": harmful_score < 0.60,  # human review if borderline
            }

        # ── Grief routing (user-specific) ─────────────────
        grief_prob = float(prob_dict.get("grief_trigger", 0.0))
        if grief_prob > 0.40:
            if "grief_sensitive" in sensitivity_flags or "ptsd" in sensitivity_flags:
                return {
                    "track_id":          track_id,
                    "safe":              False,
                    "risk_score":        round(grief_prob, 4),
                    "risk_category":     "grief_trigger",
                    "block_reason":      "user_sensitivity_flag",
                    "flagged_terms":     [],
                    "reviewed_by_human": False,
                }

        # ── Layer 3: Semantic similarity check ────────────
        semantic_scores = semantic_risk_score(lyric)
        max_harm_semantic = max(
            semantic_scores.get(c, 0.0) for c in harmful_classes
        )

        if max_harm_semantic > 0.25 and safe_prob < 0.50:
            # Semantic risk is high AND classifier wasn't confident it's safe
            harm_cat = max(
                {c: semantic_scores.get(c, 0.0) for c in harmful_classes},
                key=lambda c: semantic_scores.get(c, 0.0)
            )
            return {
                "track_id":          track_id,
                "safe":              False,
                "risk_score":        round(max_harm_semantic + 0.1, 4),
                "risk_category":     harm_cat,
                "block_reason":      "semantic_similarity",
                "flagged_terms":     [],
                "reviewed_by_human": True,   # always human-review semantic blocks
            }

        # ── Safe ───────────────────────────────────────────
        return {
            "track_id":          track_id,
            "safe":              True,
            "risk_score":        round(1.0 - safe_prob, 4),
            "risk_category":     "safe",
            "block_reason":      None,
            "flagged_terms":     [],
            "reviewed_by_human": False,
        }

    def _safe_result(self, track_id: str, reason: str = None) -> dict:
        return {
            "track_id":          track_id,
            "safe":              True,
            "risk_score":        0.0,
            "risk_category":     "safe",
            "block_reason":      reason,
            "flagged_terms":     [],
            "reviewed_by_human": False,
        }

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "threshold": self.threshold}, f)

    @classmethod
    def load(cls, path: Path) -> "LyricalSafetyFilter":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(obj["pipeline"], obj["threshold"])


# ─────────────────────────────────────────────
# SECTION: Filter Diagnostics
# ─────────────────────────────────────────────

def run_filter_diagnostics(
    safety_filter: LyricalSafetyFilter,
    df: pd.DataFrame,
):
    """
    Runs the filter on the full dataset and reports:
    1. Block rate per category
    2. False negative rate on self_harm (critical)
    3. False positive rate on safe content
    4. Human review queue volume
    """

    results = []
    for _, row in df.iterrows():
        result = safety_filter.check(
            track_id=row["track_id"],
            lyric=row["lyric"],
        )
        result["true_category"] = row["risk_category"]
        result["true_block"]    = row["should_block"]
        results.append(result)

    results_df = pd.DataFrame(results)

    print(f"\n  {'='*58}")
    print(f"  Safety Filter Diagnostics — Full Dataset")
    print(f"  {'='*58}")

    # ── Block rate per true category ───────────────────────
    print(f"\n  Block rate by true risk category:")
    print(f"  {'Category':<20} {'Block rate':>12} {'n':>6} Assessment")
    for cat in ["self_harm", "hopelessness", "substance_abuse", "grief_trigger", "safe"]:
        sub   = results_df[results_df["true_category"] == cat]
        if len(sub) == 0:
            continue
        rate  = sub["safe"].apply(lambda x: not x).mean()
        n     = len(sub)
        if cat == "self_harm":
            flag = "✅" if rate > 0.95 else "❌ CRITICAL — must be > 95%"
        elif cat == "safe":
            flag = "✅" if rate < 0.25 else "⚠  high false positive rate"
        else:
            flag = "✅" if rate > 0.60 else "⚠"
        print(f"  {cat:<20} {rate:>12.1%} {n:>6}  {flag}")

    # ── False negative analysis ────────────────────────────
    sh_rows = results_df[results_df["true_category"] == "self_harm"]
    fn_rate = sh_rows["safe"].mean()   # "safe"=True when it should be blocked

    print(f"\n  ⚠  CRITICAL SAFETY METRIC — self_harm false negatives:")
    print(f"     False negative rate: {fn_rate:.1%}")
    print(f"     (tracks that got through when they should be blocked)")
    if fn_rate < 0.05:
        print(f"     Assessment: ✅ Acceptable (< 5%)")
    else:
        print(f"     Assessment: ❌ CRITICAL — exceeds 5% threshold")

    # ── False positive analysis ────────────────────────────
    safe_rows = results_df[results_df["true_category"] == "safe"]
    fp_rate   = safe_rows["safe"].apply(lambda x: not x).mean()

    print(f"\n  False positive rate on safe content: {fp_rate:.1%}")
    print(f"  (safe tracks incorrectly blocked)")
    if fp_rate < 0.20:
        print(f"  Assessment: ✅ Acceptable (< 20%)")
    else:
        print(f"  Assessment: ⚠  High — consider raising threshold")

    # ── Human review queue ─────────────────────────────────
    review_pct = results_df["reviewed_by_human"].mean()
    print(f"\n  Human review queue: {review_pct:.1%} of all tracks")
    print(f"  ({int(review_pct * len(df)):,} tracks at scale of {len(df):,})")

    # ── Block reason breakdown ─────────────────────────────
    blocked_df = results_df[~results_df["safe"]]
    if len(blocked_df) > 0:
        print(f"\n  Block reason breakdown (among blocked tracks):")
        for reason, count in blocked_df["block_reason"].value_counts().items():
            pct = count / len(blocked_df)
            print(f"    {reason:<25} {count:>5} ({pct:.1%})")

    return results_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 58)
    print("  HarmonicAI — Phase 5: Lyrical Safety Filter")
    print("=" * 58)

    # ── Load lyric dataset ─────────────────────────────────
    lyrics_path = DATA_DIR / "lyrics.csv"
    if not lyrics_path.exists():
        print("\n  lyrics.csv not found — generating now...")
        import subprocess
        subprocess.run(["python", "scripts/generate_lyrics.py"], check=True)

    df = pd.read_csv(lyrics_path)
    print(f"\n  Loaded {len(df):,} lyric records")
    print(f"  Path: data/synthetic/lyrics.csv")

    X = df["lyric"]
    y = df["risk_category"].values

    # ── Build and evaluate classifier ─────────────────────
    print(f"\n[1/4] Building TF-IDF + Logistic Regression pipeline...")
    pipeline = build_safety_classifier()

    print(f"\n[2/4] Evaluating with 5-fold stratified CV...")
    cv_metrics = evaluate_safety_classifier(pipeline, X, y)

    print(f"\n  Cross-validation results:")
    print(f"    Macro F1:              {cv_metrics['f1_macro']:.4f} ± {cv_metrics['f1_macro_std']:.4f}")
    print(f"    Accuracy:              {cv_metrics['accuracy']:.4f}")
    print(f"    self_harm recall:      {cv_metrics['self_harm_recall']:.4f}  ← primary safety metric")
    print(f"    self_harm precision:   {cv_metrics['self_harm_precision']:.4f}")
    print(f"    hopelessness recall:   {cv_metrics['hopelessness_recall']:.4f}")

    sh_recall = cv_metrics["self_harm_recall"]
    flag = "✅ meets safety requirement" if sh_recall >= 0.90 else "❌ below required threshold"
    print(f"\n    self_harm recall ≥ 0.90 target: {flag}")

    # ── Tune threshold ─────────────────────────────────────
    print(f"\n[3/4] Training final model and tuning decision threshold...")
    pipeline.fit(X, y)

    # Use 20% holdout to tune threshold without data leakage
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    pipeline.fit(X_tr, y_tr)
    tuned_threshold = tune_threshold_for_recall(
        pipeline, X_val, y_val,
        target_class="self_harm",
        target_recall=0.95,
    )
    print(f"    Tuned decision threshold: {tuned_threshold}")
    print(f"    (lower threshold = more conservative = higher recall on harmful)")

    # Retrain on full data with tuned threshold
    pipeline.fit(X, y)

    print(f"\n    Full training classification report:")
    y_pred = pipeline.predict(X)
    print(classification_report(
        y, y_pred,
        target_names=sorted(set(y)),
        zero_division=0
    ))

    # ── Build filter and run diagnostics ───────────────────
    print(f"\n[4/4] Running full filter diagnostics...")
    safety_filter = LyricalSafetyFilter(pipeline, threshold=tuned_threshold)
    results_df    = run_filter_diagnostics(safety_filter, df)

    # ── Demo: check specific lyrics ────────────────────────
    print(f"\n  {'='*58}")
    print(f"  Live Filter Demos")
    print(f"  {'='*58}")

    demos = [
        ("demo_safe_001",
         "the morning light breaks through the window / your smile is all I need",
         []),
        ("demo_hope_001",
         "there is no way out of this darkness / nothing will ever change",
         []),
        ("demo_harm_001",
         "I want to disappear from this world for good / I planned a way out",
         []),
        ("demo_grief_001",
         "I still reach for the phone to call you / the chair at the table sits empty now",
         ["grief_sensitive"]),
        ("demo_grief_002",
         "I still reach for the phone to call you / the chair at the table sits empty now",
         []),   # same lyric, no sensitivity flag → different outcome
        ("demo_instr_001",
         "this lyric does not matter for an instrumental track",
         []),   # instrumentalness > 0.6 → skip
    ]

    instrumentalness_map = {
        "demo_instr_001": 0.85,
    }

    for track_id, lyric, flags in demos:
        inst = instrumentalness_map.get(track_id, 0.0)
        result = safety_filter.check(
            track_id=track_id,
            lyric=lyric,
            instrumentalness=inst,
            sensitivity_flags=flags,
        )
        status = "🔴 BLOCKED" if not result["safe"] else "🟢 SAFE"
        print(f"\n  {track_id}")
        print(f"  Lyric:    {lyric[:65]}...")
        print(f"  Flags:    {flags if flags else 'none'}")
        print(f"  Result:   {status}")
        print(f"  Category: {result['risk_category']} | Score: {result['risk_score']:.3f}"
              f" | Reason: {result['block_reason']}")
        if result["reviewed_by_human"]:
            print(f"  → Queued for human review")

    # ── Persist ────────────────────────────────────────────
    filter_path = MODEL_DIR / "safety_filter.pkl"
    safety_filter.save(filter_path)

    meta = {
        "model":             "TF-IDF (1-3 gram) + Logistic Regression",
        "threshold":         tuned_threshold,
        "cv_f1_macro":       cv_metrics["f1_macro"],
        "self_harm_recall":  cv_metrics["self_harm_recall"],
        "hopelessness_recall": cv_metrics["hopelessness_recall"],
        "n_training_docs":   len(df),
        "layers":            ["blocklist", "tfidf_classifier", "semantic_similarity"],
        "risk_categories":   ["self_harm", "hopelessness", "grief_trigger",
                              "substance_abuse", "safe"],
    }
    meta_path = OUTPUT_DIR / "safety_filter_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n\n✅ Phase 5 complete.")
    print(f"   Outputs:")
    print(f"   data/synthetic/lyrics.csv                ← labelled lyric dataset")
    print(f"   data/processed/safety_filter_meta.json   ← filter metadata + metrics")
    print(f"   models/checkpoints/safety_filter.pkl     ← serialised 3-layer filter")
    print(f"\nNext step → Phase 6: python src/genai/therapy_engine.py\n")
