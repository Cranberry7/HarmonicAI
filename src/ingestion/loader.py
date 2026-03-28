"""
harmonicai/src/ingestion/loader.py

Phase 1 — Data Loader & Schema Validator
=========================================
Loads the synthetic datasets and validates them against the
feature schema. In production, this module would also handle:
  - Spotify API ingestion
  - Streaming event consumption (Kafka/Kinesis)
  - Schema evolution (new features added over time)

Design principle: FAIL LOUD on schema violations.
A silent type mismatch downstream in a CNN or SVM is
catastrophic and nearly impossible to debug.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings

# ── Path Configuration ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "synthetic"

# ── Feature Contracts ────────────────────────────────────────────
# Mirrors configs/feature_schema.yaml — the single source of truth.
# If schema.yaml changes, update this dict too.

TRACK_SCHEMA = {
    "track_id":         ("object",  None,        None),
    "tempo_bpm":        ("float64", 40.0,        220.0),
    "valence":          ("float64", 0.0,         1.0),
    "energy":           ("float64", 0.0,         1.0),
    "acousticness":     ("float64", 0.0,         1.0),
    "instrumentalness": ("float64", 0.0,         1.0),
    "danceability":     ("float64", 0.0,         1.0),
    "speechiness":      ("float64", 0.0,         1.0),
    "loudness_db":      ("float64", -60.0,       0.0),
    "key":              ("int64",   0,           11),
    "mode":             ("int64",   0,           1),
    "time_signature":   ("int64",   3,           7),
    "duration_ms":      ("int64",   30_000,      3_600_000),
}

USER_SCHEMA = {
    "user_id":               ("object", None, None),
    "age_band":              ("object", None, None),
    "primary_intent":        ("object", None, None),
    "baseline_mood":         ("float64", 0.0, 100.0),
    "has_sensitivity_flag":  ("bool",   None, None),
}

SESSION_SCHEMA = {
    "session_id":      ("object",  None,  None),
    "user_id":         ("object",  None,  None),
    "track_id":        ("object",  None,  None),
    "session_intent":  ("object",  None,  None),
    "mood_pre":        ("float64", 0.0,   100.0),
    "mood_post":       ("float64", 0.0,   100.0),
    "efficacy_rating": ("int64",   1,     10),
    "completed":       ("bool",    None,  None),
    "oracle_score":    ("float64", None,  None),
    "efficacy_label":  ("int64",   -1,    1),
}

VALID_INTENTS = {
    "anxiety_relief", "sleep_induction",
    "deep_focus", "mood_uplift", "grief_processing"
}


def _validate_schema(df: pd.DataFrame, schema: dict, name: str) -> pd.DataFrame:
    """
    Validates a DataFrame against a schema contract.
    Performs type coercion where safe, raises on violations.

    Why coerce instead of reject?
    CSV round-trips lose type info (int64 → float64 for nullable columns).
    We coerce safely and loudly — you know exactly what changed.
    """
    print(f"\n  Validating {name}...")
    issues = []

    for col, (expected_type, low, high) in schema.items():

        # ── Column existence ───────────────────────────────────
        if col not in df.columns:
            issues.append(f"MISSING column: '{col}'")
            continue

        # ── Type coercion ──────────────────────────────────────
        actual_type = str(df[col].dtype)
        if actual_type != expected_type:
            try:
                if expected_type == "bool":
                    df[col] = df[col].astype(bool)
                elif expected_type == "int64":
                    df[col] = df[col].astype("int64")
                elif expected_type == "float64":
                    df[col] = df[col].astype("float64")
                print(f"    ℹ Coerced '{col}': {actual_type} → {expected_type}")
            except Exception as e:
                issues.append(f"TYPE ERROR on '{col}': cannot coerce {actual_type} → {expected_type}: {e}")
                continue

        # ── Null check ─────────────────────────────────────────
        null_count = df[col].isnull().sum()
        if null_count > 0:
            issues.append(f"NULL VALUES: '{col}' has {null_count} nulls")

        # ── Range check ────────────────────────────────────────
        if low is not None and high is not None and expected_type != "object":
            out_of_range = ((df[col] < low) | (df[col] > high)).sum()
            if out_of_range > 0:
                issues.append(
                    f"RANGE VIOLATION: '{col}' has {out_of_range} values "
                    f"outside [{low}, {high}]"
                )

    if issues:
        msg = f"\n❌ Schema violations in {name}:\n" + "\n".join(f"   • {i}" for i in issues)
        raise ValueError(msg)

    print(f"    ✅ {name}: {len(df):,} rows, {len(df.columns)} columns — schema valid")
    return df


def load_tracks() -> pd.DataFrame:
    path = DATA_DIR / "tracks.csv"
    df = pd.read_csv(path)
    return _validate_schema(df, TRACK_SCHEMA, "tracks")


def load_users() -> pd.DataFrame:
    path = DATA_DIR / "users.csv"
    df = pd.read_csv(path)
    return _validate_schema(df, USER_SCHEMA, "users")


def load_sessions() -> pd.DataFrame:
    path = DATA_DIR / "sessions.csv"
    df = pd.read_csv(path)
    return _validate_schema(df, SESSION_SCHEMA, "sessions")


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Master loader. Returns (tracks, users, sessions).
    Use this as the entry point for all downstream phases.
    """
    print("=" * 55)
    print("  HarmonicAI — Phase 1: Data Loader")
    print("=" * 55)
    tracks   = load_tracks()
    users    = load_users()
    sessions = load_sessions()

    # ── Intent validation (cross-field) ───────────────────────
    bad_intents = set(sessions["session_intent"].unique()) - VALID_INTENTS
    if bad_intents:
        raise ValueError(f"Unknown session_intent values: {bad_intents}")

    print(f"\n  ✅ All datasets loaded and validated.")
    return tracks, users, sessions


if __name__ == "__main__":
    tracks, users, sessions = load_all()
    print(f"\n  tracks.shape:   {tracks.shape}")
    print(f"  users.shape:    {users.shape}")
    print(f"  sessions.shape: {sessions.shape}")
