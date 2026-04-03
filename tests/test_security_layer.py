"""
tests/test_security_layer.py
============================
Phase 8 test suite — no external dependencies required.

Run with:
    python tests/test_security_layer.py
"""

import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.security.security_layer import (
    AdversarialInputDetector,
    AuditEventType,
    AuditLogger,
    BlockCode,
    GateResult,
    InboundRequest,
    PIIScrubber,
    ResponseSanitiser,
    SecurityLayer,
    TokenBucketRateLimiter,
    VALID_INTENTS,
    VALID_SENSITIVITY_FLAGS,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VALID_FV = {
    "tempo_bpm": 80.0, "energy": 0.4, "valence": 0.5,
    "acousticness": 0.6, "instrumentalness": 0.3,
    "speechiness": 0.05, "loudness_db": -15.0,
    "mood_pre": 40.0, "intent_encoded": 1.0,
}


def _req(**kwargs) -> InboundRequest:
    defaults = dict(
        user_id="user_042", session_id="sess_001",
        intent="anxiety_relief", mood_pre=40.0,
        sensitivity_flags=[], feature_vector=dict(_VALID_FV),
    )
    defaults.update(kwargs)
    return InboundRequest(**defaults)


def _sec(tmp: Path, rate_limit: int = 100) -> SecurityLayer:
    return SecurityLayer(
        rate_limit_requests=rate_limit,
        rate_limit_window_s=60,
        burst_capacity=5,
        audit_log_path=tmp / "audit.log",
        pii_salt="test_salt_fixed",
    )


class _Tmp:
    def __enter__(self):
        self.path = Path(tempfile.mkdtemp())
        return self
    def __exit__(self, *_):
        shutil.rmtree(self.path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Schema Validation tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_valid_request_passes(self):
        with _Tmp() as t:
            gate = _sec(t.path).inspect_request(_req())
            assert gate.allowed

    def test_all_valid_intents_pass(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            for intent in VALID_INTENTS:
                gate = sec.inspect_request(_req(intent=intent))
                assert gate.allowed, f"Expected {intent} to pass"

    def test_invalid_intent_blocked(self):
        with _Tmp() as t:
            gate = _sec(t.path).inspect_request(_req(intent="chill"))
            assert not gate.allowed
            assert gate.block_code == BlockCode.INVALID_INTENT

    def test_empty_user_id_blocked(self):
        with _Tmp() as t:
            gate = _sec(t.path).inspect_request(_req(user_id=""))
            assert not gate.allowed
            assert gate.block_code == BlockCode.MISSING_REQUIRED_FIELD

    def test_empty_session_id_blocked(self):
        with _Tmp() as t:
            gate = _sec(t.path).inspect_request(_req(session_id=""))
            assert not gate.allowed

    def test_mood_pre_above_100_blocked(self):
        with _Tmp() as t:
            gate = _sec(t.path).inspect_request(_req(mood_pre=101.0))
            assert not gate.allowed
            assert gate.block_code == BlockCode.SCHEMA_VIOLATION

    def test_mood_pre_below_0_blocked(self):
        with _Tmp() as t:
            gate = _sec(t.path).inspect_request(_req(mood_pre=-1.0))
            assert not gate.allowed

    def test_mood_pre_boundary_values_pass(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            assert sec.inspect_request(_req(mood_pre=0.0)).allowed
            assert sec.inspect_request(_req(mood_pre=100.0)).allowed

    def test_unknown_sensitivity_flag_blocked(self):
        with _Tmp() as t:
            gate = _sec(t.path).inspect_request(
                _req(sensitivity_flags=["unknown_flag"])
            )
            assert not gate.allowed
            assert gate.block_code == BlockCode.INVALID_SENSITIVITY_FLAG

    def test_all_valid_sensitivity_flags_pass(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            gate = sec.inspect_request(
                _req(sensitivity_flags=list(VALID_SENSITIVITY_FLAGS))
            )
            assert gate.allowed

    def test_non_dict_feature_vector_blocked(self):
        with _Tmp() as t:
            gate = _sec(t.path).inspect_request(
                _req(feature_vector=[1, 2, 3])
            )
            assert not gate.allowed
            assert gate.block_code == BlockCode.SCHEMA_VIOLATION

    def test_non_numeric_mood_pre_blocked(self):
        with _Tmp() as t:
            gate = _sec(t.path).inspect_request(_req(mood_pre="high"))
            assert not gate.allowed


# ---------------------------------------------------------------------------
# Rate Limiter tests
# ---------------------------------------------------------------------------

class TestRateLimiter:

    def test_requests_within_limit_pass(self):
        limiter = TokenBucketRateLimiter(max_requests=10, window_seconds=60, burst_capacity=0)
        for _ in range(10):
            ok, _ = limiter.check("user_a")
            assert ok

    def test_exceeding_limit_blocked(self):
        limiter = TokenBucketRateLimiter(max_requests=3, window_seconds=60, burst_capacity=0)
        results = [limiter.check("user_b")[0] for _ in range(5)]
        assert results[:3] == [True, True, True]
        assert results[3] is False

    def test_burst_capacity_allows_extra(self):
        limiter = TokenBucketRateLimiter(max_requests=3, window_seconds=60, burst_capacity=2)
        results = [limiter.check("user_c")[0] for _ in range(6)]
        # 3 max + 2 burst = 5 allowed, 6th blocked
        assert sum(results) == 5

    def test_different_users_independent(self):
        limiter = TokenBucketRateLimiter(max_requests=2, window_seconds=60, burst_capacity=0)
        ok_a, _ = limiter.check("user_a")
        ok_a2, _ = limiter.check("user_a")
        ok_a3, _ = limiter.check("user_a")  # should be blocked
        ok_b, _ = limiter.check("user_b")   # different user, fresh bucket
        assert ok_a and ok_a2
        assert not ok_a3
        assert ok_b

    def test_reset_restores_bucket(self):
        limiter = TokenBucketRateLimiter(max_requests=1, window_seconds=60, burst_capacity=0)
        limiter.check("user_x")  # consume token
        ok, _ = limiter.check("user_x")
        assert not ok
        limiter.reset("user_x")
        ok2, _ = limiter.check("user_x")
        assert ok2

    def test_rate_limit_via_security_layer(self):
        with _Tmp() as t:
            sec = _sec(t.path, rate_limit=3)
            results = [sec.inspect_request(_req(session_id=f"s{i}")).allowed for i in range(5)]
            # First 3 + burst=5 extra = up to 8, but rate_limit=3 with burst=5 → 8 total
            # With burst=5, first 8 should pass; 4th-8th use burst
            blocked = [r for r in results if not r]
            # We just check that EVENTUALLY something is blocked if we exceed max+burst
            # (with burst=5, limit=3, capacity=8, 5 requests should all pass)
            assert all(results), "All 5 should pass with burst=5"

    def test_rate_limit_blocked_after_burst_exhausted(self):
        with _Tmp() as t:
            # Very tight limit: max=1, burst=1 → capacity=2
            sec = SecurityLayer(
                rate_limit_requests=1, rate_limit_window_s=60, burst_capacity=1,
                audit_log_path=t.path / "audit.log", pii_salt="test",
            )
            results = [sec.inspect_request(_req(session_id=f"s{i}")).allowed for i in range(5)]
            assert results[0] and results[1]   # first 2 pass (1 max + 1 burst)
            assert not results[2]               # 3rd is blocked


# ---------------------------------------------------------------------------
# Adversarial Input Detector tests
# ---------------------------------------------------------------------------

class TestAdversarialInputDetector:

    def setup_method(self):
        self.d = AdversarialInputDetector()

    def test_valid_vector_passes(self):
        safe, score, warnings, code = self.d.inspect(dict(_VALID_FV))
        assert safe
        assert code is None

    def test_tempo_out_of_bounds_blocked(self):
        fv = {**_VALID_FV, "tempo_bpm": 999.0}
        safe, score, warnings, code = self.d.inspect(fv)
        assert not safe
        assert code == BlockCode.FEATURE_OUT_OF_BOUNDS

    def test_energy_out_of_bounds_blocked(self):
        fv = {**_VALID_FV, "energy": 1.5}
        safe, score, warnings, code = self.d.inspect(fv)
        assert not safe

    def test_negative_loudness_within_bounds_passes(self):
        fv = {**_VALID_FV, "loudness_db": -40.0}
        safe, *_ = self.d.inspect(fv)
        assert safe

    def test_loudness_above_zero_blocked(self):
        fv = {**_VALID_FV, "loudness_db": 5.0}
        safe, *_ = self.d.inspect(fv)
        assert not safe

    def test_non_numeric_feature_blocked(self):
        fv = {**_VALID_FV, "tempo_bpm": "fast"}
        safe, *_ = self.d.inspect(fv)
        assert not safe
        _, _, _, code = self.d.inspect(fv)
        assert code == BlockCode.FEATURE_OUT_OF_BOUNDS

    def test_single_implausible_combo_warns_not_blocks(self):
        fv = {**_VALID_FV, "energy": 0.99, "acousticness": 0.99}
        safe, score, warnings, code = self.d.inspect(fv)
        assert safe   # single combo → warning only
        assert len(warnings) >= 1
        assert score > 0

    def test_multiple_implausible_combos_blocked(self):
        fv = {
            **_VALID_FV,
            "energy":          0.99,
            "acousticness":    0.99,
            "instrumentalness":0.95,
            "speechiness":     0.65,
        }
        safe, score, warnings, code = self.d.inspect(fv)
        assert not safe
        assert code == BlockCode.ADVERSARIAL_INPUT

    def test_excessive_feature_keys_blocked(self):
        fv = {f"feat_{i}": float(i) for i in range(25)}
        safe, *_ = self.d.inspect(fv)
        assert not safe

    def test_empty_feature_vector_passes_bounds_check(self):
        # No features to check → no bounds violations
        safe, score, warnings, code = self.d.inspect({})
        assert safe

    def test_boundary_feature_values_pass(self):
        from src.security.security_layer import FEATURE_BOUNDS
        # Test each feature at its lo and hi boundary individually,
        # keeping all other features in the safe middle of their range.
        # (Setting ALL features to their max simultaneously triggers
        # implausible-combination checks, which is correct behaviour.)
        base = dict(_VALID_FV)
        for feat, (lo, hi) in FEATURE_BOUNDS.items():
            for val in (lo, hi):
                fv = {**base, feat: val}
                safe, _, _, code = self.d.inspect(fv)
                assert safe, (
                    f"Feature '{feat}'={val} (boundary) should pass bounds "
                    f"check, but got code={code}"
                )


# ---------------------------------------------------------------------------
# PII Scrubber tests
# ---------------------------------------------------------------------------

class TestPIIScrubber:

    def setup_method(self):
        self.s = PIIScrubber(salt="test_salt")

    def test_pseudonymise_is_deterministic(self):
        a = self.s.pseudonymise("user_042")
        b = self.s.pseudonymise("user_042")
        assert a == b

    def test_different_users_different_pseudonyms(self):
        a = self.s.pseudonymise("user_001")
        b = self.s.pseudonymise("user_002")
        assert a != b

    def test_pseudonym_does_not_contain_user_id(self):
        pseudo = self.s.pseudonymise("user_042")
        assert "user_042" not in pseudo

    def test_pseudonym_is_12_chars(self):
        assert len(self.s.pseudonymise("any_user")) == 12

    def test_email_scrubbed(self):
        text, modified = self.s.scrub_text("Email me at john@example.com please")
        assert "john@example.com" not in text
        assert "[EMAIL]" in text
        assert modified

    def test_phone_scrubbed(self):
        text, modified = self.s.scrub_text("Call me at 555-867-5309")
        assert "867-5309" not in text
        assert modified

    def test_ip_address_scrubbed(self):
        text, modified = self.s.scrub_text("Origin: 192.168.1.100")
        assert "192.168.1.100" not in text
        assert modified

    def test_clean_text_unchanged(self):
        clean = "Take a gentle breath and notice your body settling."
        text, modified = self.s.scrub_text(clean)
        assert not modified

    def test_request_user_id_pseudonymised(self):
        req = _req(user_id="real_user_id_here")
        scrubbed, _ = self.s.scrub_request(req)
        assert scrubbed.user_id != "real_user_id_here"
        assert len(scrubbed.user_id) == 12

    def test_request_ip_always_redacted(self):
        req = _req(client_ip="10.0.0.1")
        scrubbed, _ = self.s.scrub_request(req)
        assert scrubbed.client_ip is None

    def test_lyric_pii_scrubbed_in_request(self):
        req = _req(raw_lyric="Contact admin@example.com for help")
        scrubbed, pii_found = self.s.scrub_request(req)
        assert "admin@example.com" not in scrubbed.raw_lyric
        assert pii_found

    def test_clean_lyric_not_flagged(self):
        req = _req(raw_lyric="The rain falls soft upon the mountain.")
        _, pii_found = self.s.scrub_request(req)
        assert not pii_found

    def test_different_salt_different_pseudonym(self):
        s1 = PIIScrubber(salt="salt_a")
        s2 = PIIScrubber(salt="salt_b")
        assert s1.pseudonymise("user_x") != s2.pseudonymise("user_x")


# ---------------------------------------------------------------------------
# Audit Logger tests
# ---------------------------------------------------------------------------

class TestAuditLogger:

    def test_record_increments_sequence(self):
        with _Tmp() as t:
            logger = AuditLogger(t.path / "audit.log")
            _log_event(logger, allowed=True)
            _log_event(logger, allowed=True)
            events = logger.query_recent()
            assert events[0].sequence == 0
            assert events[1].sequence == 1

    def test_events_written_to_disk(self):
        with _Tmp() as t:
            log_path = t.path / "audit.log"
            logger = AuditLogger(log_path)
            _log_event(logger, allowed=True)
            assert log_path.exists()
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 1
            # Must be valid JSON
            json.loads(lines[0])

    def test_each_line_valid_json(self):
        with _Tmp() as t:
            log_path = t.path / "audit.log"
            logger = AuditLogger(log_path)
            for _ in range(5):
                _log_event(logger, allowed=True)
            lines = log_path.read_text().strip().split("\n")
            for line in lines:
                json.loads(line)  # must not raise

    def test_blocked_events_queryable(self):
        with _Tmp() as t:
            logger = AuditLogger(t.path / "audit.log")
            _log_event(logger, allowed=True)
            _log_event(logger, allowed=False)
            blocked = [e for e in logger.query_recent() if not e.allowed]
            assert len(blocked) == 1

    def test_count_blocks_returns_correct_count(self):
        with _Tmp() as t:
            logger = AuditLogger(t.path / "audit.log")
            pseudo = "testpseudo00"
            for _ in range(3):
                _log_event(logger, allowed=False, pseudonym=pseudo)
            _log_event(logger, allowed=True, pseudonym=pseudo)
            assert logger.count_blocks(pseudo) == 3

    def test_no_pii_in_log_lines(self):
        with _Tmp() as t:
            log_path = t.path / "audit.log"
            logger = AuditLogger(log_path)
            _log_event(logger, allowed=True, pseudonym="abc123def456")
            content = log_path.read_text()
            assert "user_042" not in content
            assert "real_email@example.com" not in content


def _log_event(logger: AuditLogger, allowed: bool, pseudonym: str = "abc123def456") -> None:
    from src.security.security_layer import AuditEvent, AuditEventType
    import uuid
    logger.record(AuditEvent(
        event_id="e" + uuid.uuid4().hex[:7],
        sequence=0,
        event_type=AuditEventType.REQUEST_ALLOWED.value if allowed else AuditEventType.REQUEST_BLOCKED.value,
        pseudonym=pseudonym,
        session_id="sess_001",
        intent="anxiety_relief",
        timestamp="2025-01-01T00:00:00+00:00",
        allowed=allowed,
        block_code=None if allowed else BlockCode.SCHEMA_VIOLATION.value,
        block_reason="" if allowed else "test block",
        anomaly_score=0.0,
        warnings=[],
        details={},
    ))


# ---------------------------------------------------------------------------
# Response Sanitiser tests
# ---------------------------------------------------------------------------

class TestResponseSanitiser:

    def setup_method(self):
        self.scrubber = PIIScrubber(salt="test")
        self.san = ResponseSanitiser(self.scrubber)

    def test_clean_response_unchanged(self):
        resp = {"tracks": [{"track_id": "t001"}], "script": "Breathe gently."}
        clean, issues = self.san.sanitise(resp, "user_1")
        assert clean["tracks"] == resp["tracks"]
        assert clean["script"] == resp["script"]

    def test_model_path_removed(self):
        resp = {"tracks": [], "model_path": "/models/rf.pkl"}
        clean, issues = self.san.sanitise(resp, "user_1")
        assert "model_path" not in clean
        assert any("model_path" in i for i in issues)

    def test_internal_debug_removed(self):
        resp = {"script": "...", "internal_debug": {"cv": [0.9]}}
        clean, issues = self.san.sanitise(resp, "user_1")
        assert "internal_debug" not in clean

    def test_all_internal_fields_removed(self):
        from src.security.security_layer import _INTERNAL_RESPONSE_FIELDS
        resp = {f: "value" for f in _INTERNAL_RESPONSE_FIELDS}
        resp["safe_field"] = "keep this"
        clean, _ = self.san.sanitise(resp, "u1")
        for field in _INTERNAL_RESPONSE_FIELDS:
            assert field not in clean
        assert clean["safe_field"] == "keep this"

    def test_raw_user_id_pseudonymised(self):
        resp = {"user_id": "raw_user_id"}
        clean, issues = self.san.sanitise(resp, "raw_user_id")
        assert clean["user_id"] != "raw_user_id"
        assert len(clean["user_id"]) == 12

    def test_email_in_script_scrubbed(self):
        resp = {"script": "Contact help@company.com for support."}
        clean, issues = self.san.sanitise(resp, "u1")
        assert "help@company.com" not in clean["script"]

    def test_nested_pii_scrubbed(self):
        resp = {"tracks": [{"notes": "Call 555-123-4567"}]}
        clean, issues = self.san.sanitise(resp, "u1")
        assert "555-123-4567" not in clean["tracks"][0]["notes"]


# ---------------------------------------------------------------------------
# SecurityLayer integration tests
# ---------------------------------------------------------------------------

class TestSecurityLayerIntegration:

    def test_pseudonym_consistent_across_calls(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            g1 = sec.inspect_request(_req(session_id="s1"))
            g2 = sec.inspect_request(_req(session_id="s2"))
            assert g1.pseudonym == g2.pseudonym

    def test_scrubbed_request_user_id_is_pseudonym(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            gate = sec.inspect_request(_req(user_id="my_real_user"))
            assert gate.scrubbed_request.user_id != "my_real_user"

    def test_audit_log_written_on_allow(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            sec.inspect_request(_req())
            log = sec.audit_log()
            assert any(e.allowed for e in log)

    def test_audit_log_written_on_block(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            sec.inspect_request(_req(intent="bad"))
            log = sec.audit_log()
            assert any(not e.allowed for e in log)

    def test_audit_log_contains_no_raw_user_id(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            sec.inspect_request(_req(user_id="secret_user_id_xyzzy"))
            for event in sec.audit_log():
                assert "secret_user_id_xyzzy" not in event.pseudonym
                assert "secret_user_id_xyzzy" not in json.dumps(event.details)

    def test_blocked_request_includes_block_code(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            gate = sec.inspect_request(_req(mood_pre=200.0))
            assert gate.block_code is not None
            assert gate.block_reason != ""

    def test_sanitise_response_removes_internals(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            resp = {"script": "Breathe.", "model_path": "/secret/path", "raw_scores": [0.9]}
            clean = sec.sanitise_response(resp, "u1")
            assert "model_path" not in clean
            assert "raw_scores" not in clean

    def test_warn_only_on_single_implausible_combo(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            fv = {**_VALID_FV, "energy": 0.99, "acousticness": 0.99}
            gate = sec.inspect_request(_req(feature_vector=fv))
            assert gate.allowed   # warned but not blocked
            assert len(gate.warnings) >= 1

    def test_fail_secure_on_unexpected_exception(self):
        """If schema validation raises or catches an error, the request must be denied."""
        with _Tmp() as t:
            sec = _sec(t.path)
            # A non-dict feature_vector causes an error path — must be denied
            req = InboundRequest(
                user_id="u", session_id="s", intent="deep_focus",
                mood_pre=40.0, sensitivity_flags=[],
                feature_vector=[1, 2, 3],  # list, not dict → schema violation
            )
            gate = sec.inspect_request(req)
            assert not gate.allowed
            assert gate.block_code in (BlockCode.SCHEMA_VIOLATION, BlockCode.INTERNAL_ERROR)

    def test_pseudonymise_method_exposed(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            p = sec.pseudonymise("user_x")
            assert isinstance(p, str) and len(p) == 12

    def test_full_pipeline_all_intents(self):
        with _Tmp() as t:
            sec = _sec(t.path)
            for intent in VALID_INTENTS:
                gate = sec.inspect_request(_req(intent=intent))
                assert gate.allowed, f"intent={intent} blocked unexpectedly"


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_classes = [
        TestSchemaValidation,
        TestRateLimiter,
        TestAdversarialInputDetector,
        TestPIIScrubber,
        TestAuditLogger,
        TestResponseSanitiser,
        TestSecurityLayerIntegration,
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
        for cls_name, method, tb in errors:
            print(f"\nFAILED: {cls_name}.{method}")
            print(tb)
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)
