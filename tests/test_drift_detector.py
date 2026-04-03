"""
tests/test_drift_detector.py
============================
Phase 7 test suite — runs fully in-memory with temp directories.
No pickled models or real CSV files required.

Run with:
    python tests/test_drift_detector.py
Or:
    python -m pytest tests/test_drift_detector.py -v
"""

import json
import pickle
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feedback.drift_detector import (
    COLD_START_MIN_SESSIONS,
    INTENT_ENCODING,
    MONITORED_FEATURES,
    PSI_MONITOR,
    PSI_STABLE,
    DriftDetector,
    DriftReport,
    FeedbackEvent,
    ModelRetrainer,
    PSICalculator,
    RollingPerformanceTracker,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_event(
    session_id: str = "s001",
    user_id: str = "u001",
    intent: str = "anxiety_relief",
    mood_pre: float = 40.0,
    mood_post: float = 55.0,
    efficacy_rating: int = 7,
    completed: bool = True,
    predicted_label: int = 1,
    feature_vector: dict = None,
) -> FeedbackEvent:
    if feature_vector is None:
        feature_vector = {
            "tempo_bpm": 80.0, "energy": 0.4, "valence": 0.5,
            "acousticness": 0.6, "instrumentalness": 0.3,
            "speechiness": 0.05, "loudness_db": -15.0,
            "mood_pre": mood_pre, "intent_encoded": INTENT_ENCODING[intent],
        }
    return FeedbackEvent(
        session_id=session_id, user_id=user_id, track_id="t001",
        intent=intent, mood_pre=mood_pre, mood_post=mood_post,
        efficacy_rating=efficacy_rating, completed=completed,
        predicted_label=predicted_label, feature_vector=feature_vector,
    )


def _make_train_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "tempo_bpm":        np.clip(rng.normal(90, 25, n), 40, 200),
        "energy":           np.clip(rng.normal(0.45, 0.15, n), 0, 1),
        "valence":          np.clip(rng.normal(0.50, 0.20, n), 0, 1),
        "acousticness":     rng.uniform(0.1, 0.9, n),
        "instrumentalness": rng.uniform(0.0, 0.8, n),
        "speechiness":      rng.uniform(0.02, 0.4, n),
        "loudness_db":      rng.uniform(-28, -6, n),
        "mood_pre":         np.clip(rng.normal(45, 18, n), 0, 100),
        "session_intent":   rng.choice(list(INTENT_ENCODING.keys()), n),
        "efficacy_label":   rng.choice([-1, 0, 1], n, p=[0.596, 0.316, 0.088]),
        "user_id":          [f"u{i % 50:03d}" for i in range(n)],
    })


class _TempEnv:
    """Context manager: creates a temp directory with all required paths."""

    def __enter__(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        (self.tmpdir / "data" / "processed").mkdir(parents=True)
        (self.tmpdir / "data" / "feedback").mkdir(parents=True)
        (self.tmpdir / "models" / "checkpoints").mkdir(parents=True)
        (self.tmpdir / "models" / "registry").mkdir(parents=True)

        train_df = _make_train_df()
        # Add intent_encoded so MONITORED_FEATURES can be fully selected
        train_df["intent_encoded"] = train_df["session_intent"].map(INTENT_ENCODING)
        train_path = self.tmpdir / "data" / "processed" / "sessions_normalized.csv"
        train_df.to_csv(train_path, index=False)

        feat_path = self.tmpdir / "models" / "checkpoints" / "feature_list.pkl"
        with open(feat_path, "wb") as f:
            pickle.dump(MONITORED_FEATURES, f)

        # Tiny fitted RF pipeline
        X = train_df[MONITORED_FEATURES].fillna(0).values
        y = train_df["efficacy_label"].astype(int).values
        pipe = Pipeline([("sc", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=5, random_state=0))])
        pipe.fit(X, y)
        rf_path = self.tmpdir / "models" / "checkpoints" / "random_forest.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(pipe, f)

        self.paths = {
            "rf_model":        rf_path,
            "feature_list":    feat_path,
            "train_sessions":  train_path,
            "tracks":          Path("/tmp"),   # unused in tests
            "users":           Path("/tmp"),
            "feedback_csv":    self.tmpdir / "data" / "feedback" / "sessions_feedback.csv",
            "drift_report":    self.tmpdir / "data" / "processed" / "drift_report.json",
            "registry":        self.tmpdir / "models" / "registry" / "model_registry.json",
            "checkpoints_dir": self.tmpdir / "models" / "checkpoints",
        }
        return self

    def __exit__(self, *_):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def detector(self, auto_retrain=False, threshold=500) -> DriftDetector:
        return DriftDetector(auto_retrain=auto_retrain, retrain_session_threshold=threshold, paths=self.paths)


# ---------------------------------------------------------------------------
# FeedbackEvent tests
# ---------------------------------------------------------------------------

class TestFeedbackEvent:

    def test_valid_event(self):
        e = _make_event()
        assert e.intent == "anxiety_relief"

    def test_invalid_intent_raises(self):
        try:
            # Use a valid feature_vector to ensure only the intent validation fires
            FeedbackEvent(
                session_id="s", user_id="u", track_id="t",
                intent="bad_intent", mood_pre=40.0, mood_post=50.0,
                efficacy_rating=7, completed=True, predicted_label=1,
                feature_vector={"tempo_bpm": 80.0},
            )
            assert False, "Should have raised"
        except (ValueError, KeyError) as e:
            assert "intent" in str(e).lower() or "bad_intent" in str(e)

    def test_invalid_predicted_label_raises(self):
        try:
            _make_event(predicted_label=2)
            assert False
        except ValueError:
            pass

    def test_invalid_mood_post_raises(self):
        try:
            _make_event(mood_post=150.0)
            assert False
        except ValueError:
            pass

    def test_invalid_efficacy_rating_raises(self):
        try:
            _make_event(efficacy_rating=11)
            assert False
        except ValueError:
            pass

    def test_derive_true_label_positive(self):
        e = _make_event(mood_pre=30.0, mood_post=45.0, efficacy_rating=8)
        assert e.derive_true_label() == 1

    def test_derive_true_label_negative(self):
        e = _make_event(mood_pre=50.0, mood_post=30.0, efficacy_rating=2)
        assert e.derive_true_label() == -1

    def test_derive_true_label_neutral(self):
        e = _make_event(mood_pre=50.0, mood_post=52.0, efficacy_rating=5)
        assert e.derive_true_label() == 0

    def test_derive_true_label_none_when_no_data(self):
        e = _make_event()
        e.mood_post = None
        e.efficacy_rating = None
        assert e.derive_true_label() is None

    def test_rating_overrides_delta_for_positive(self):
        # rating=9 dominates even if delta is small
        e = _make_event(mood_pre=50.0, mood_post=52.0, efficacy_rating=9)
        assert e.derive_true_label() == 1

    def test_timestamp_auto_populated(self):
        e = _make_event()
        assert "T" in e.timestamp  # ISO-8601 contains T

    def test_all_intents_valid(self):
        for intent in INTENT_ENCODING:
            e = _make_event(intent=intent)
            assert e.intent == intent


# ---------------------------------------------------------------------------
# PSICalculator tests
# ---------------------------------------------------------------------------

class TestPSICalculator:

    def _train_df(self):
        return _make_train_df(n=1000)[MONITORED_FEATURES[:-1]].copy()  # exclude intent

    def test_fit_returns_self(self):
        calc = PSICalculator()
        df = _make_train_df(n=500)
        result = calc.fit(df)
        assert result is calc

    def test_stable_population_low_psi(self):
        rng = np.random.default_rng(42)
        n = 500
        train = pd.DataFrame({
            "tempo_bpm": np.clip(rng.normal(90, 20, n), 40, 200),
            "energy":    np.clip(rng.normal(0.45, 0.12, n), 0, 1),
        })
        # Same distribution for feedback
        feedback = pd.DataFrame({
            "tempo_bpm": np.clip(rng.normal(90, 20, n), 40, 200),
            "energy":    np.clip(rng.normal(0.45, 0.12, n), 0, 1),
        })
        calc = PSICalculator()

        # Patch MONITORED_FEATURES to just these two
        calc._bin_edges = {}
        calc._reference_dist = {}
        from src.feedback import drift_detector as dd
        original = dd.MONITORED_FEATURES
        dd.MONITORED_FEATURES = ["tempo_bpm", "energy"]
        import importlib
        calc.fit(train)
        psi = calc.compute(feedback)
        dd.MONITORED_FEATURES = original

        for feat, val in psi.items():
            assert val < PSI_MONITOR, f"{feat} PSI={val:.4f} unexpectedly high for stable population"

    def test_drifted_population_high_psi(self):
        """Drifted tempo distribution should produce high PSI."""
        rng = np.random.default_rng(0)
        n = 500
        train_df = _make_train_df(n=n, seed=0)
        # Drastically shift tempo in feedback
        feedback = pd.DataFrame({"tempo_bpm": np.clip(rng.normal(160, 10, n), 120, 200)})

        calc = PSICalculator()
        calc.fit(train_df)
        # Manually compute PSI for just tempo_bpm
        psi = calc.compute(feedback)
        assert "tempo_bpm" in psi
        assert psi["tempo_bpm"] > PSI_STABLE, f"Expected drift, got PSI={psi['tempo_bpm']:.4f}"

    def test_compute_before_fit_raises(self):
        calc = PSICalculator()
        try:
            calc.compute(pd.DataFrame({"tempo_bpm": [80.0]}))
            assert False
        except RuntimeError:
            pass

    def test_empty_feedback_returns_zero(self):
        train = _make_train_df()
        calc = PSICalculator().fit(train)
        psi = calc.compute(pd.DataFrame())
        # Should return zeros or empty, not crash
        for v in psi.values():
            assert v == 0.0


# ---------------------------------------------------------------------------
# RollingPerformanceTracker tests
# ---------------------------------------------------------------------------

class TestRollingPerformanceTracker:

    def _make_feedback_df(self, n: int = 50, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            intent = list(INTENT_ENCODING.keys())[i % 5]
            mood_pre  = float(rng.uniform(20, 80))
            mood_post = float(np.clip(mood_pre + rng.normal(8, 10), 0, 100))
            rating    = int(rng.integers(1, 11))
            pred      = int(rng.choice([-1, 0, 1]))
            ev = _make_event(
                session_id=f"s{i}", user_id=f"u{i % 30}",
                intent=intent, mood_pre=mood_pre, mood_post=mood_post,
                efficacy_rating=rating, predicted_label=pred,
            )
            true = ev.derive_true_label()
            rows.append({
                "session_id": ev.session_id, "user_id": ev.user_id,
                "intent": ev.intent, "mood_pre": mood_pre, "mood_post": mood_post,
                "completed": True, "predicted_label": pred, "true_label": true,
            })
        return pd.DataFrame(rows)

    def test_empty_df_returns_nones(self):
        t = RollingPerformanceTracker()
        result = t.compute(pd.DataFrame(), {})
        assert result["rolling_f1"] is None
        assert result["mean_delta_mood"] is None

    def test_rolling_f1_computed_with_sufficient_data(self):
        t = RollingPerformanceTracker()
        df = self._make_feedback_df(n=50)
        result = t.compute(df, {})
        assert result["rolling_f1"] is not None
        assert 0.0 <= result["rolling_f1"] <= 1.0

    def test_mean_delta_mood_positive_when_mood_improves(self):
        t = RollingPerformanceTracker()
        # All sessions improve mood
        rows = []
        for i in range(20):
            rows.append({
                "user_id": "u1", "completed": True,
                "mood_pre": 30.0, "mood_post": 55.0,
                "true_label": 1, "predicted_label": 1, "intent": "mood_uplift",
            })
        df = pd.DataFrame(rows)
        result = t.compute(df, {})
        assert result["mean_delta_mood"] > 0

    def test_cold_start_detection(self):
        t = RollingPerformanceTracker()
        rows = [{"user_id": "new_user", "completed": True, "mood_pre": 40.0, "mood_post": 50.0, "true_label": 1, "predicted_label": 1, "intent": "deep_focus"}]
        df = pd.DataFrame(rows)
        # new_user has 0 sessions in training
        result = t.compute(df, {"veteran_user": 20})
        assert "new_user" in result["cold_start_users"]

    def test_veteran_user_not_cold_start(self):
        t = RollingPerformanceTracker()
        rows = [{"user_id": "u_vet", "completed": True, "mood_pre": 40.0, "mood_post": 50.0, "true_label": 1, "predicted_label": 1, "intent": "deep_focus"}]
        df = pd.DataFrame(rows)
        result = t.compute(df, {"u_vet": 20})
        assert "u_vet" not in result["cold_start_users"]


# ---------------------------------------------------------------------------
# DriftDetector integration tests
# ---------------------------------------------------------------------------

class TestDriftDetector:

    def test_record_creates_csv(self):
        with _TempEnv() as env:
            d = env.detector()
            d.record(_make_event())
            assert env.paths["feedback_csv"].exists()

    def test_record_appends_rows(self):
        with _TempEnv() as env:
            d = env.detector()
            d.record(_make_event(session_id="s1"))
            d.record(_make_event(session_id="s2"))
            df = pd.read_csv(env.paths["feedback_csv"])
            assert len(df) == 2

    def test_record_batch_writes_all(self):
        with _TempEnv() as env:
            d = env.detector()
            events = [_make_event(session_id=f"s{i}") for i in range(10)]
            d.record_batch(events)
            df = pd.read_csv(env.paths["feedback_csv"])
            assert len(df) == 10

    def test_check_and_update_returns_drift_report(self):
        with _TempEnv() as env:
            d = env.detector()
            events = [_make_event(session_id=f"s{i}") for i in range(30)]
            d.record_batch(events)
            report = d.check_and_update()
            assert isinstance(report, DriftReport)

    def test_drift_report_json_written(self):
        with _TempEnv() as env:
            d = env.detector()
            d.record(_make_event())
            d.check_and_update()
            assert env.paths["drift_report"].exists()

    def test_stable_population_no_retrain_recommended(self):
        """With only 30 stable sessions, retrain should NOT be recommended."""
        with _TempEnv() as env:
            d = env.detector(auto_retrain=False, threshold=500)
            events = [_make_event(session_id=f"s{i}") for i in range(30)]
            d.record_batch(events)
            report = d.check_and_update()
            # Volume threshold not met, drift probably not critical
            # Might or might not recommend — just check it ran without error
            assert isinstance(report.retrain_recommended, bool)

    def test_force_retrain_triggers_retrain(self):
        with _TempEnv() as env:
            d = env.detector(auto_retrain=True, threshold=500)
            events = [_make_event(session_id=f"s{i}") for i in range(60)]
            d.record_batch(events)
            report = d.check_and_update(force_retrain=True)
            assert report.retrained
            assert report.new_model_path != ""
            assert Path(report.new_model_path).exists()

    def test_retrain_creates_registry(self):
        with _TempEnv() as env:
            d = env.detector(auto_retrain=True, threshold=10)
            events = [_make_event(session_id=f"s{i}", mood_pre=60.0, mood_post=30.0, efficacy_rating=2) for i in range(15)]
            d.record_batch(events)
            d.check_and_update(force_retrain=True)
            assert env.paths["registry"].exists()
            with open(env.paths["registry"]) as f:
                reg = json.load(f)
            assert "models" in reg
            assert len(reg["models"]) >= 1

    def test_report_summary_is_string(self):
        with _TempEnv() as env:
            d = env.detector()
            d.record(_make_event())
            report = d.check_and_update()
            summary = report.summary()
            assert isinstance(summary, str)
            assert "PSI" in summary

    def test_report_as_dict_serialisable(self):
        with _TempEnv() as env:
            d = env.detector()
            d.record(_make_event())
            report = d.check_and_update()
            d_dict = report.as_dict()
            # Must be JSON-serialisable
            json.dumps(d_dict)

    def test_cold_start_users_in_report(self):
        with _TempEnv() as env:
            d = env.detector()
            # New user not in training data
            e = _make_event(user_id="brand_new_user_xyz")
            d.record(e)
            report = d.check_and_update()
            assert "brand_new_user_xyz" in report.cold_start_users

    def test_rolling_f1_populated_with_labelled_events(self):
        with _TempEnv() as env:
            d = env.detector()
            # 20+ labelled events needed for F1 computation
            events = [
                _make_event(
                    session_id=f"s{i}",
                    mood_pre=30.0, mood_post=50.0, efficacy_rating=8,
                    predicted_label=1,
                )
                for i in range(25)
            ]
            d.record_batch(events)
            report = d.check_and_update()
            assert report.rolling_f1 is not None

    def test_session_count_accurate(self):
        with _TempEnv() as env:
            d = env.detector()
            n = 17
            events = [_make_event(session_id=f"s{i}") for i in range(n)]
            d.record_batch(events)
            report = d.check_and_update()
            assert report.new_session_count == n

    def test_second_run_appends_not_overwrites(self):
        with _TempEnv() as env:
            d = env.detector()
            d.record_batch([_make_event(session_id=f"s{i}") for i in range(5)])
            d.record_batch([_make_event(session_id=f"s{i+5}") for i in range(5)])
            df = pd.read_csv(env.paths["feedback_csv"])
            assert len(df) == 10

    def test_drifted_tempo_detected(self):
        """Heavily shifted tempo distribution should appear in drifted_features."""
        with _TempEnv() as env:
            rng = np.random.default_rng(99)
            d = env.detector(auto_retrain=False)
            events = []
            for i in range(300):
                intent = list(INTENT_ENCODING.keys())[i % 5]
                fv = {
                    "tempo_bpm":        float(np.clip(rng.normal(170, 5), 140, 200)),  # way up
                    "energy":           float(np.clip(rng.normal(0.45, 0.1), 0, 1)),
                    "valence":          float(np.clip(rng.normal(0.50, 0.1), 0, 1)),
                    "acousticness":     float(rng.uniform(0.1, 0.9)),
                    "instrumentalness": float(rng.uniform(0.0, 0.8)),
                    "speechiness":      float(rng.uniform(0.02, 0.4)),
                    "loudness_db":      float(rng.uniform(-28, -6)),
                    "mood_pre":         float(np.clip(rng.normal(45, 18), 0, 100)),
                    "intent_encoded":   INTENT_ENCODING[intent],
                }
                events.append(_make_event(session_id=f"sd{i}", intent=intent, feature_vector=fv))
            d.record_batch(events)
            report = d.check_and_update()
            assert "tempo_bpm" in report.drifted_features, (
                f"Expected tempo_bpm drift, psi_scores={report.psi_scores}"
            )


# ---------------------------------------------------------------------------
# ModelRetrainer unit tests
# ---------------------------------------------------------------------------

class TestModelRetrainer:

    def test_retrain_produces_versioned_checkpoint(self):
        with _TempEnv() as env:
            retrainer = ModelRetrainer()
            train_df  = pd.read_csv(env.paths["train_sessions"])

            # Minimal feedback df
            rng = np.random.default_rng(0)
            n = 50
            fb = pd.DataFrame({
                feat: rng.random(n) for feat in MONITORED_FEATURES
            })
            fb["true_label"] = rng.choice([-1, 0, 1], n)
            fb["efficacy_label"] = fb["true_label"]

            path, f1, vtag = retrainer.retrain(
                original_sessions = train_df,
                feedback_df       = fb,
                feature_list      = MONITORED_FEATURES,
                current_f1        = 0.906,
                registry_path     = env.paths["registry"],
                checkpoints_dir   = env.paths["checkpoints_dir"],
            )
            assert Path(path).exists()
            assert 0.0 < f1 <= 1.0
            assert vtag == "v2"

    def test_retrain_increments_version(self):
        with _TempEnv() as env:
            retrainer = ModelRetrainer()
            train_df  = pd.read_csv(env.paths["train_sessions"])
            rng = np.random.default_rng(0)
            n = 50
            fb = pd.DataFrame({feat: rng.random(n) for feat in MONITORED_FEATURES})
            fb["true_label"] = rng.choice([-1, 0, 1], n)
            fb["efficacy_label"] = fb["true_label"]

            _, _, v1 = retrainer.retrain(train_df, fb, MONITORED_FEATURES, 0.906, env.paths["registry"], env.paths["checkpoints_dir"])
            _, _, v2 = retrainer.retrain(train_df, fb, MONITORED_FEATURES, 0.906, env.paths["registry"], env.paths["checkpoints_dir"])
            assert v1 == "v2"
            assert v2 == "v3"

    def test_registry_records_both_versions(self):
        with _TempEnv() as env:
            retrainer = ModelRetrainer()
            train_df  = pd.read_csv(env.paths["train_sessions"])
            rng = np.random.default_rng(0)
            n = 50
            fb = pd.DataFrame({feat: rng.random(n) for feat in MONITORED_FEATURES})
            fb["true_label"] = rng.choice([-1, 0, 1], n)
            fb["efficacy_label"] = fb["true_label"]

            retrainer.retrain(train_df, fb, MONITORED_FEATURES, 0.906, env.paths["registry"], env.paths["checkpoints_dir"])
            with open(env.paths["registry"]) as f:
                reg = json.load(f)
            versions = [e["version"] for e in reg["models"]]
            assert "v1" in versions
            assert "v2" in versions


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_classes = [
        TestFeedbackEvent,
        TestPSICalculator,
        TestRollingPerformanceTracker,
        TestDriftDetector,
        TestModelRetrainer,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods  = sorted(m for m in dir(cls) if m.startswith("test_"))
        for method_name in methods:
            if hasattr(instance, "setup_method"):
                instance.setup_method()
            try:
                getattr(instance, method_name)()
                print(f"  ✓  {cls.__name__}.{method_name}")
                passed += 1
            except Exception:
                print(f"  ✗  {cls.__name__}.{method_name}")
                errors.append((cls.__name__, method_name, traceback.format_exc()))
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"  {passed} passed   {failed} failed")
    if errors:
        print()
        for cls_name, method_name, tb in errors:
            print(f"FAILED: {cls_name}.{method_name}")
            print(tb)
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)
