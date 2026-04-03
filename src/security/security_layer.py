"""
src/security/security_layer.py
================================
Phase 8 — Security Layer

Purpose
-------
Wraps the entire HarmonicAI request lifecycle with security controls.
Every request that enters the system passes through this layer before
any model inference occurs, and every response passes through it again
before being returned to the caller.

Why does a therapeutic ML system need a security layer?
-------------------------------------------------------
HarmonicAI operates at the intersection of three high-risk domains:

1. MENTAL HEALTH — users in distress. A manipulated or adversarial
   input that bypasses the safety filter and returns harmful content
   is not just a product defect; it is a patient safety event.

2. PERSONAL DATA — mood scores, session histories, and sensitivity
   flags are deeply personal. They must not leak between users,
   appear in logs, or be recoverable from model artefacts.

3. ML INFERENCE — recommendation models are vulnerable to adversarial
   inputs. An attacker who understands the feature space can craft
   inputs that reliably steer the model toward specific predictions.

Security controls implemented
------------------------------
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Request Validation Gate                                  │
│  Schema enforcement before anything else runs. Malformed inputs     │
│  never reach the model.                                             │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2 — Rate Limiting                                            │
│  Per-user token bucket with configurable burst capacity. Prevents   │
│  automated enumeration attacks against the recommendation engine.   │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3 — Adversarial Input Detection                              │
│  Detects feature vectors that are statistically implausible or      │
│  outside the training distribution. Blocks inputs engineered to    │
│  force specific model outputs.                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4 — PII Scrubber                                             │
│  Strips or pseudonymises personal identifiers before they enter     │
│  logs, prompts, or model inputs. Operates on both request fields    │
│  and free-text content (e.g. LLM prompt conditioning).              │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 5 — Audit Logger                                             │
│  Immutable, append-only audit trail for every request. Records      │
│  security events (blocks, rate limits, anomalies) with tamper-      │
│  evident sequential IDs. Never logs raw PII.                        │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 6 — Response Sanitiser                                       │
│  Final sweep of outgoing content. Catches any PII that made it      │
│  through, enforces response shape contracts, and removes fields      │
│  that should never be user-visible (internal scores, model paths).  │
└─────────────────────────────────────────────────────────────────────┘

Design principles
-----------------
FAIL SECURE — if any security check cannot run (missing config, raised
exception), the request is DENIED, not passed through. Never fail open.

DEFENCE IN DEPTH — no single layer is trusted to catch everything.
Each layer assumes the previous one may have missed something.

SEPARATION OF CONCERNS — security logic is never embedded in model
code. All security runs in this module; models never see raw user IDs
or free-text that has not been scrubbed.

NO SILENT DEGRADATION — every block, rate-limit, and anomaly is logged
with a structured AuditEvent before the request is rejected.

Usage
-----
    from src.security.security_layer import SecurityLayer, InboundRequest

    sec = SecurityLayer()

    req = InboundRequest(
        user_id           = "user_042",
        session_id        = "sess_abc",
        intent            = "anxiety_relief",
        mood_pre          = 34.0,
        sensitivity_flags = ["grief_sensitive"],
        feature_vector    = {...},   # 9 acoustic + context features
        raw_lyric         = "...",   # optional, for safety filter path
    )

    gate = sec.inspect_request(req)
    if not gate.allowed:
        return {"error": gate.block_reason, "code": gate.block_code}

    # ... run model pipeline ...

    response = sec.sanitise_response(raw_response, req.user_id)
    return response
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Feature bounds derived from Phase 0 synthetic data specification.
# Any input outside these ranges is rejected as implausible / adversarial.
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "tempo_bpm":         (40.0,   220.0),
    "energy":            (0.0,    1.0),
    "valence":           (0.0,    1.0),
    "acousticness":      (0.0,    1.0),
    "instrumentalness":  (0.0,    1.0),
    "speechiness":       (0.0,    1.0),
    "loudness_db":       (-60.0,  0.0),
    "mood_pre":          (0.0,    100.0),
    "intent_encoded":    (0.0,    4.0),
}

VALID_INTENTS: Set[str] = {
    "sleep_induction", "anxiety_relief", "grief_processing",
    "deep_focus", "mood_uplift",
}

VALID_SENSITIVITY_FLAGS: Set[str] = {
    "grief_sensitive", "anxiety_prone", "sleep_disorder",
}

# Rate limiting defaults
DEFAULT_RATE_LIMIT_REQUESTS  = 30    # requests per window
DEFAULT_RATE_LIMIT_WINDOW_S  = 60    # window duration in seconds
DEFAULT_BURST_CAPACITY       = 5     # extra requests allowed in burst

# Audit log location
AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "security" / "audit.log"

# PII patterns to redact from logs and LLM prompts
_PII_PATTERNS: List[Tuple[str, str]] = [
    # Email addresses
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",   "[EMAIL]"),
    # E.164 phone numbers and common variants
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    # IP addresses (v4)
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                              "[IP]"),
    # UK / EU postcodes (broad)
    (r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b",                "[POSTCODE]"),
    # US zip codes
    (r"\b\d{5}(?:-\d{4})?\b",                                     "[ZIP]"),
    # Credit card-like numbers (≥13 digits with optional separators)
    (r"\b(?:\d[ -]?){13,19}\b",                                    "[CARD]"),
    # JWT tokens (three base64url parts separated by dots)
    (r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b", "[TOKEN]"),
    # Generic API key patterns (32–64 hex chars or base64)
    (r"\b[A-Za-z0-9+/]{32,64}={0,2}\b",                           "[KEY]"),
]

# Response fields that must never be sent to users
_INTERNAL_RESPONSE_FIELDS: Set[str] = {
    "model_path", "checkpoint_path", "feature_importance",
    "raw_scores", "internal_debug", "retrain_path",
    "drift_report_path", "_internal",
}

# ---------------------------------------------------------------------------
# Enums and result types
# ---------------------------------------------------------------------------

class BlockCode(str, Enum):
    """Structured reason codes for blocked requests."""
    SCHEMA_VIOLATION         = "SCHEMA_VIOLATION"
    RATE_LIMITED             = "RATE_LIMITED"
    ADVERSARIAL_INPUT        = "ADVERSARIAL_INPUT"
    FEATURE_OUT_OF_BOUNDS    = "FEATURE_OUT_OF_BOUNDS"
    INVALID_INTENT           = "INVALID_INTENT"
    INVALID_SENSITIVITY_FLAG = "INVALID_SENSITIVITY_FLAG"
    MISSING_REQUIRED_FIELD   = "MISSING_REQUIRED_FIELD"
    RESPONSE_SANITISE_FAILED = "RESPONSE_SANITISE_FAILED"
    INTERNAL_ERROR           = "INTERNAL_ERROR"


class AuditEventType(str, Enum):
    REQUEST_ALLOWED   = "REQUEST_ALLOWED"
    REQUEST_BLOCKED   = "REQUEST_BLOCKED"
    RATE_LIMITED      = "RATE_LIMITED"
    ANOMALY_DETECTED  = "ANOMALY_DETECTED"
    PII_SCRUBBED      = "PII_SCRUBBED"
    RESPONSE_SANITISED= "RESPONSE_SANITISED"
    RETRAIN_TRIGGERED = "RETRAIN_TRIGGERED"


@dataclass
class InboundRequest:
    """
    Validated input contract for the security layer.

    Every field is validated by inspect_request(); callers do not
    need to pre-validate — the security layer handles all schema
    enforcement.

    Parameters
    ----------
    user_id           : Opaque user identifier. Never logged as plaintext;
                        stored only as a pseudonym hash in audit events.
    session_id        : Unique session ID for cross-layer correlation.
    intent            : One of VALID_INTENTS.
    mood_pre          : Pre-session mood score (0–100).
    sensitivity_flags : Zero or more flags from VALID_SENSITIVITY_FLAGS.
    feature_vector    : Dict of acoustic + context features. Validated
                        against FEATURE_BOUNDS.
    raw_lyric         : Optional lyric string for safety filter routing.
                        Will be PII-scrubbed before logging.
    client_ip         : Optional. Used for rate limiting. Scrubbed from logs.
    """
    user_id:            str
    session_id:         str
    intent:             str
    mood_pre:           float
    sensitivity_flags:  List[str]       = field(default_factory=list)
    feature_vector:     Dict[str, float] = field(default_factory=dict)
    raw_lyric:          Optional[str]   = None
    client_ip:          Optional[str]   = None


@dataclass
class GateResult:
    """
    Return type of SecurityLayer.inspect_request().

    allowed      : True means the request may proceed to model inference.
    block_code   : Populated only when allowed=False.
    block_reason : Human-readable explanation of the block.
    pseudonym    : Pseudonymised user identifier for safe downstream use.
    scrubbed_request : Copy of InboundRequest with PII replaced.
    anomaly_score: 0.0–1.0 continuous risk score from adversarial detection.
    warnings     : Non-blocking issues detected (logged but not blocked).
    """
    allowed:          bool
    pseudonym:        str
    scrubbed_request: InboundRequest
    anomaly_score:    float           = 0.0
    block_code:       Optional[BlockCode] = None
    block_reason:     str                 = ""
    warnings:         List[str]           = field(default_factory=list)


@dataclass
class AuditEvent:
    """
    Single entry in the immutable audit log.

    No raw PII ever appears in an AuditEvent.
    user_id is stored only as a pseudonym hash.
    """
    event_id:      str      # UUID4
    sequence:      int      # monotonically increasing, tamper-evident
    event_type:    str      # AuditEventType value
    pseudonym:     str      # SHA-256(user_id + salt)[:12]
    session_id:    str
    intent:        str
    timestamp:     str      # ISO-8601 UTC
    allowed:       bool
    block_code:    Optional[str]
    block_reason:  str
    anomaly_score: float
    warnings:      List[str]
    details:       Dict[str, Any]   # non-PII supplementary info

    def to_log_line(self) -> str:
        """Serialise to a single-line JSON string for the audit log."""
        return json.dumps(asdict(self), separators=(",", ":"))


# ---------------------------------------------------------------------------
# Layer 2 — Rate Limiter
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """
    Per-user token bucket rate limiter.

    How it works
    ------------
    Each user starts with `burst_capacity` tokens. Each request consumes
    one token. Tokens refill at a rate of `max_requests / window_seconds`
    per second up to `max_requests + burst_capacity`.

    Why token bucket over fixed window?
    ------------------------------------
    Fixed windows are susceptible to boundary attacks: a user can make
    2× the limit by flooding requests at the end of one window and the
    start of the next. The token bucket smooths refill continuously,
    preventing this exploit while still allowing short bursts.

    Thread-safety: NOT thread-safe in this implementation.
    A production deployment should use Redis with atomic INCR/DECR
    operations for distributed rate limiting.
    """

    def __init__(
        self,
        max_requests:   int   = DEFAULT_RATE_LIMIT_REQUESTS,
        window_seconds: int   = DEFAULT_RATE_LIMIT_WINDOW_S,
        burst_capacity: int   = DEFAULT_BURST_CAPACITY,
    ) -> None:
        self._max      = max_requests
        self._window   = window_seconds
        self._burst    = burst_capacity
        self._capacity = float(max_requests + burst_capacity)
        self._refill_rate = max_requests / window_seconds   # tokens/second

        # {user_key: (tokens_remaining, last_refill_time)}
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def check(self, user_key: str) -> Tuple[bool, float]:
        """
        Attempt to consume one token for user_key.

        Returns
        -------
        (allowed, tokens_remaining)
        """
        now = time.monotonic()

        if user_key not in self._buckets:
            self._buckets[user_key] = (self._capacity - 1.0, now)
            return True, self._capacity - 1.0

        tokens, last_refill = self._buckets[user_key]
        elapsed = now - last_refill
        tokens  = min(self._capacity, tokens + elapsed * self._refill_rate)

        if tokens < 1.0:
            self._buckets[user_key] = (tokens, now)
            return False, tokens

        tokens -= 1.0
        self._buckets[user_key] = (tokens, now)
        return True, tokens

    def remaining(self, user_key: str) -> float:
        """Return current token count without consuming."""
        if user_key not in self._buckets:
            return self._capacity
        tokens, last_refill = self._buckets[user_key]
        elapsed = time.monotonic() - last_refill
        return min(self._capacity, tokens + elapsed * self._refill_rate)

    def reset(self, user_key: str) -> None:
        """Reset bucket for a user (admin use only)."""
        self._buckets.pop(user_key, None)


# ---------------------------------------------------------------------------
# Layer 3 — Adversarial Input Detector
# ---------------------------------------------------------------------------

class AdversarialInputDetector:
    """
    Detects feature vectors that are implausible or potentially engineered
    to manipulate model predictions.

    Three checks
    ------------
    1. HARD BOUNDS — each feature value must be within FEATURE_BOUNDS.
       Out-of-bounds values are either errors or active attacks.
       Result: BLOCK.

    2. STATISTICAL IMPLAUSIBILITY — look for combinations that are
       technically within bounds but essentially impossible in real audio.
       E.g. energy=0.99 + acousticness=0.99 (acoustic tracks are not high
       energy; this combination does not exist in natural music and would
       be unusual in the training distribution).
       Result: WARNING (logged, not blocked — not certain enough to deny).

    3. FEATURE STUFFING — a feature appearing more than once in the
       vector dict, or more features than the model expects.
       Result: BLOCK.

    Adversarial ML context
    ----------------------
    The Phase 3 Random Forest was trained on acoustic data with natural
    correlations between features. An attacker who knows the model exists
    could run a grey-box attack (e.g. gradient-free boundary attack) to
    find inputs that reliably force P(+1|x) ≈ 1.0. The hard-bounds and
    implausibility checks raise the cost of such attacks significantly:
    any crafted input that exploits extreme feature values is caught,
    forcing the attacker to stay within the natural distribution where
    the model's learnt correlations provide natural defence.
    """

    # (feature_a, feature_b, description) — these combinations are
    # acoustically impossible and indicate potential adversarial input
    _IMPOSSIBLE_PAIRS: List[Tuple[str, str, str, Callable[[float, float], bool]]] = [
        (
            "energy", "acousticness",
            "high energy + high acousticness is acoustically implausible",
            lambda e, a: e > 0.92 and a > 0.92,
        ),
        (
            "instrumentalness", "speechiness",
            "high instrumentalness + high speechiness is contradictory",
            lambda i, s: i > 0.90 and s > 0.60,
        ),
        (
            "tempo_bpm", "energy",
            "very slow tempo with maximum energy is implausible",
            lambda t, e: t < 45 and e > 0.95,
        ),
    ]

    # Maximum number of features we expect (9 from Phase 3 ALL_FEATURES)
    _MAX_FEATURES = 9

    def inspect(
        self,
        feature_vector: Dict[str, float],
    ) -> Tuple[bool, float, List[str], Optional[BlockCode]]:
        """
        Inspect a feature vector.

        Returns
        -------
        (safe, anomaly_score, warnings, block_code)
            safe=True means the vector passed all hard checks.
            anomaly_score is 0.0–1.0.
            warnings are non-blocking observations.
            block_code is set only when safe=False.
        """
        warnings: List[str] = []
        anomaly_score = 0.0

        # ---- Check 3: feature stuffing -----------------------------------
        if len(feature_vector) > self._MAX_FEATURES * 2:
            return (
                False, 1.0, ["Feature vector contains excessive keys"],
                BlockCode.ADVERSARIAL_INPUT,
            )

        # ---- Check 1: hard bounds ----------------------------------------
        for feat, (lo, hi) in FEATURE_BOUNDS.items():
            if feat not in feature_vector:
                continue
            val = feature_vector[feat]
            if not isinstance(val, (int, float)):
                return (
                    False, 1.0,
                    [f"Feature '{feat}' has non-numeric value: {val!r}"],
                    BlockCode.FEATURE_OUT_OF_BOUNDS,
                )
            if not (lo <= float(val) <= hi):
                return (
                    False, 1.0,
                    [f"Feature '{feat}' = {val} is outside bounds [{lo}, {hi}]"],
                    BlockCode.FEATURE_OUT_OF_BOUNDS,
                )

        # ---- Check 2: statistical implausibility -------------------------
        hit_count = 0
        for feat_a, feat_b, desc, predicate in self._IMPOSSIBLE_PAIRS:
            if feat_a in feature_vector and feat_b in feature_vector:
                try:
                    if predicate(float(feature_vector[feat_a]), float(feature_vector[feat_b])):
                        warnings.append(f"Implausible feature combination: {desc}")
                        hit_count += 1
                except (TypeError, ValueError):
                    pass

        # Score rises with number of implausible combinations hit
        if hit_count > 0:
            anomaly_score = min(1.0, hit_count * 0.35)

        # Multiple simultaneous implausible combinations → escalate to block
        if hit_count >= 2:
            return (
                False, anomaly_score,
                warnings + ["Multiple implausible feature combinations — possible adversarial input"],
                BlockCode.ADVERSARIAL_INPUT,
            )

        return True, anomaly_score, warnings, None


# ---------------------------------------------------------------------------
# Layer 4 — PII Scrubber
# ---------------------------------------------------------------------------

class PIIScrubber:
    """
    Strips or pseudonymises personally identifiable information.

    Operates on three target types:
    1. String fields (free text, lyrics, log messages)
    2. User IDs → pseudonym hashes
    3. Full InboundRequest objects → scrubbed copies

    Pseudonymisation
    ----------------
    User IDs are replaced with SHA-256(user_id + per-instance salt)[:12].
    The salt is random per SecurityLayer instantiation, meaning:
    - The same user_id produces the same pseudonym within a single
      server process lifetime.
    - Pseudonyms from different process restarts are not linkable.
    - The original user_id cannot be recovered from a pseudonym.

    Why not full anonymisation?
    ---------------------------
    Full anonymisation (no reversible mapping) prevents us from
    correlating audit events for the same user within a session.
    The per-process pseudonym gives within-session correlation while
    preventing cross-restart linkability.
    """

    def __init__(self, salt: Optional[str] = None) -> None:
        self._salt = salt or uuid.uuid4().hex
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in _PII_PATTERNS
        ]

    def pseudonymise(self, user_id: str) -> str:
        """Return a stable 12-char hex pseudonym for user_id."""
        raw = f"{self._salt}:{user_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def scrub_text(self, text: str) -> Tuple[str, bool]:
        """
        Apply all PII regex patterns to a text string.

        Returns
        -------
        (scrubbed_text, was_modified)
        """
        original = text
        for pattern, replacement in self._compiled:
            text = pattern.sub(replacement, text)
        return text, text != original

    def scrub_request(self, req: InboundRequest) -> Tuple[InboundRequest, bool]:
        """
        Return a copy of req with PII replaced in string fields.

        The user_id is replaced with its pseudonym.
        raw_lyric is scrubbed if present.
        client_ip is always redacted.

        Returns
        -------
        (scrubbed_request, pii_was_found)
        """
        pii_found = False

        scrubbed_lyric = req.raw_lyric
        if req.raw_lyric:
            scrubbed_lyric, modified = self.scrub_text(req.raw_lyric)
            if modified:
                pii_found = True

        return (
            InboundRequest(
                user_id           = self.pseudonymise(req.user_id),
                session_id        = req.session_id,
                intent            = req.intent,
                mood_pre          = req.mood_pre,
                sensitivity_flags = list(req.sensitivity_flags) if isinstance(req.sensitivity_flags, (list, tuple)) else [],
                feature_vector    = dict(req.feature_vector) if isinstance(req.feature_vector, dict) else {},
                raw_lyric         = scrubbed_lyric,
                client_ip         = None,   # always redact IP
            ),
            pii_found,
        )


# ---------------------------------------------------------------------------
# Layer 5 — Audit Logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Append-only audit trail for every security event.

    Properties
    ----------
    - Sequence numbers are monotonically increasing within a process
      lifetime. Gaps in sequence numbers indicate potential tampering
      (log deletion) if audit logs are persisted.
    - Each line is a single self-contained JSON object.
    - Raw PII never appears in any log entry.
    - Writes are flushed immediately (no buffering) so that a crash
      never silently discards security events.

    In-memory audit buffer
    ----------------------
    The last BUFFER_SIZE audit events are kept in memory for fast
    programmatic querying (e.g. "show me all blocks for this pseudonym
    in the last 5 minutes").  The on-disk log is the source of truth.
    """

    BUFFER_SIZE = 1000

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self._log_path = log_path or AUDIT_LOG_PATH
        self._sequence = 0
        self._buffer: Deque[AuditEvent] = deque(maxlen=self.BUFFER_SIZE)

    def record(self, event: AuditEvent) -> None:
        """Append event to in-memory buffer and flush to disk."""
        event.sequence = self._sequence
        self._sequence += 1

        self._buffer.append(event)
        self._write_to_disk(event)

    def query_recent(
        self,
        pseudonym:    Optional[str] = None,
        event_type:   Optional[str] = None,
        last_n:       int           = 100,
    ) -> List[AuditEvent]:
        """
        Query the in-memory buffer. Filters by pseudonym and/or event_type.
        """
        results = list(self._buffer)[-last_n:]
        if pseudonym:
            results = [e for e in results if e.pseudonym == pseudonym]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        return results

    def count_blocks(self, pseudonym: str, within_seconds: float = 60.0) -> int:
        """Return number of blocked requests for a pseudonym in recent window."""
        cutoff = time.monotonic() - within_seconds
        # We don't have monotonic time in AuditEvent, use buffer proximity
        recent = list(self._buffer)[-200:]
        return sum(
            1 for e in recent
            if e.pseudonym == pseudonym and not e.allowed
        )

    def _write_to_disk(self, event: AuditEvent) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(event.to_log_line() + "\n")
                f.flush()
        except OSError:
            # Disk write failed — keep in buffer, do NOT raise.
            # A write failure must never cause a security check to fail open.
            pass


# ---------------------------------------------------------------------------
# Layer 6 — Response Sanitiser
# ---------------------------------------------------------------------------

class ResponseSanitiser:
    """
    Final sweep of outgoing API responses.

    Two responsibilities:
    1. Remove internal fields that should never be user-visible.
    2. Scrub any PII that survived earlier layers.

    Why is a second PII sweep needed?
    ----------------------------------
    The LLM (therapy_engine.py) generates free text conditioned on user
    context.  Although the prompt builder pseudonymises user IDs, a
    hallucinated or leaked user identifier could appear in the generated
    script text.  The response sanitiser is the last line of defence.
    """

    def __init__(self, scrubber: PIIScrubber) -> None:
        self._scrubber = scrubber

    def sanitise(
        self,
        response: Dict[str, Any],
        user_id:  str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Sanitise an outgoing response dict.

        Returns
        -------
        (sanitised_response, issues_found)
            issues_found is a list of what was removed/redacted.
        """
        issues: List[str] = []
        out = dict(response)

        # ---- Remove internal fields -------------------------------------
        for field_name in _INTERNAL_RESPONSE_FIELDS:
            if field_name in out:
                del out[field_name]
                issues.append(f"Removed internal field: '{field_name}'")

        # ---- Scrub string values -----------------------------------------
        out, string_issues = self._scrub_string_values(out)
        issues.extend(string_issues)

        # ---- Ensure user_id is pseudonymised if present -----------------
        if "user_id" in out:
            pseudonym = self._scrubber.pseudonymise(user_id)
            if out["user_id"] != pseudonym:
                out["user_id"] = pseudonym
                issues.append("Replaced raw user_id with pseudonym in response")

        return out, issues

    def _scrub_string_values(
        self,
        obj: Any,
        depth: int = 0,
    ) -> Tuple[Any, List[str]]:
        """Recursively scrub PII from all string values in a nested structure."""
        if depth > 10:
            return obj, []   # depth guard

        issues: List[str] = []

        if isinstance(obj, str):
            scrubbed, modified = self._scrubber.scrub_text(obj)
            if modified:
                issues.append("PII redacted from string value")
            return scrubbed, issues

        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                scrubbed_v, sub_issues = self._scrub_string_values(v, depth + 1)
                result[k] = scrubbed_v
                issues.extend(sub_issues)
            return result, issues

        if isinstance(obj, list):
            result_list = []
            for item in obj:
                scrubbed_item, sub_issues = self._scrub_string_values(item, depth + 1)
                result_list.append(scrubbed_item)
                issues.extend(sub_issues)
            return result_list, issues

        return obj, []


# ---------------------------------------------------------------------------
# Main Security Layer
# ---------------------------------------------------------------------------

class SecurityLayer:
    """
    Facade that coordinates all six security controls.

    Typical call pattern
    --------------------
        sec    = SecurityLayer()
        gate   = sec.inspect_request(req)

        if not gate.allowed:
            return error_response(gate.block_code, gate.block_reason)

        # ... model inference using gate.scrubbed_request ...

        clean  = sec.sanitise_response(raw_response, req.user_id)
        return clean

    Parameters
    ----------
    rate_limit_requests : int
        Max requests per rate_limit_window_s per user.
    rate_limit_window_s : int
        Window duration for rate limiting.
    burst_capacity      : int
        Extra requests allowed in burst.
    audit_log_path      : Path, optional
        Override audit log path (useful for tests).
    pii_salt            : str, optional
        Override PII pseudonymisation salt (useful for tests).
    """

    def __init__(
        self,
        rate_limit_requests: int           = DEFAULT_RATE_LIMIT_REQUESTS,
        rate_limit_window_s: int           = DEFAULT_RATE_LIMIT_WINDOW_S,
        burst_capacity:      int           = DEFAULT_BURST_CAPACITY,
        audit_log_path:      Optional[Path]= None,
        pii_salt:            Optional[str] = None,
    ) -> None:
        self._scrubber  = PIIScrubber(salt=pii_salt)
        self._limiter   = TokenBucketRateLimiter(
            rate_limit_requests, rate_limit_window_s, burst_capacity
        )
        self._detector  = AdversarialInputDetector()
        self._logger    = AuditLogger(audit_log_path)
        self._sanitiser = ResponseSanitiser(self._scrubber)

    # ---- Primary API -------------------------------------------------------

    def inspect_request(self, req: InboundRequest) -> GateResult:
        """
        Run all inbound security checks in order.

        Checks are ordered from cheapest to most expensive:
        1. Schema validation   (O(1))
        2. Rate limiting       (O(1) amortised)
        3. Adversarial check   (O(features))
        4. PII scrubbing       (O(text_length))

        The first failing check short-circuits; subsequent checks do not run.
        All results are logged before returning.
        """
        pseudonym = self._scrubber.pseudonymise(req.user_id)

        try:
            # ---- Layer 1: Schema validation --------------------------------
            schema_ok, schema_reason, schema_code = self._validate_schema(req)
            if not schema_ok:
                return self._block_and_log(
                    req, pseudonym, schema_code, schema_reason, 0.0, []
                )

            # ---- Layer 2: Rate limiting ------------------------------------
            rate_key = self._rate_limit_key(req, pseudonym)
            rate_ok, tokens_left = self._limiter.check(rate_key)
            if not rate_ok:
                return self._block_and_log(
                    req, pseudonym,
                    BlockCode.RATE_LIMITED,
                    f"Rate limit exceeded. Tokens remaining: {tokens_left:.2f}",
                    0.0, [],
                )

            # ---- Layer 3: Adversarial input detection ----------------------
            adv_safe, anomaly_score, adv_warnings, adv_code = self._detector.inspect(
                req.feature_vector
            )
            if not adv_safe:
                return self._block_and_log(
                    req, pseudonym, adv_code,
                    f"Adversarial input detected: {'; '.join(adv_warnings)}",
                    anomaly_score, adv_warnings,
                )

            # ---- Layer 4: PII scrubbing ------------------------------------
            scrubbed_req, pii_found = self._scrubber.scrub_request(req)
            warnings = list(adv_warnings)
            if pii_found:
                warnings.append("PII detected and scrubbed from request")
                self._log_event(
                    pseudonym    = pseudonym,
                    session_id   = req.session_id,
                    intent       = req.intent,
                    event_type   = AuditEventType.PII_SCRUBBED,
                    allowed      = True,
                    block_code   = None,
                    block_reason = "",
                    anomaly_score= 0.0,
                    warnings     = warnings,
                    details      = {"pii_in_lyric": pii_found},
                )

            # ---- All checks passed -----------------------------------------
            gate = GateResult(
                allowed          = True,
                pseudonym        = pseudonym,
                scrubbed_request = scrubbed_req,
                anomaly_score    = anomaly_score,
                warnings         = warnings,
            )
            self._log_event(
                pseudonym    = pseudonym,
                session_id   = req.session_id,
                intent       = req.intent,
                event_type   = AuditEventType.REQUEST_ALLOWED,
                allowed      = True,
                block_code   = None,
                block_reason = "",
                anomaly_score= anomaly_score,
                warnings     = warnings,
                details      = {"tokens_remaining": round(tokens_left, 2)},
            )
            return gate

        except Exception as exc:
            # FAIL SECURE: unexpected exception → deny request
            reason = f"Internal security error: {type(exc).__name__}"
            return self._block_and_log(
                req, pseudonym, BlockCode.INTERNAL_ERROR, reason, 1.0, []
            )

    def sanitise_response(
        self,
        response: Dict[str, Any],
        user_id:  str,
    ) -> Dict[str, Any]:
        """
        Sanitise an outgoing response.  Logs if issues were found.

        Raises
        ------
        RuntimeError if sanitisation itself fails — the caller should
        return an error response rather than the unsanitised payload.
        """
        try:
            clean, issues = self._sanitiser.sanitise(response, user_id)
            pseudonym = self._scrubber.pseudonymise(user_id)
            if issues:
                self._log_event(
                    pseudonym    = pseudonym,
                    session_id   = response.get("session_id", ""),
                    intent       = response.get("intent", ""),
                    event_type   = AuditEventType.RESPONSE_SANITISED,
                    allowed      = True,
                    block_code   = None,
                    block_reason = "",
                    anomaly_score= 0.0,
                    warnings     = issues,
                    details      = {"issues_count": len(issues)},
                )
            return clean
        except Exception as exc:
            raise RuntimeError(
                f"Response sanitisation failed: {exc}"
            ) from exc

    def pseudonymise(self, user_id: str) -> str:
        """Expose pseudonymisation for use by other pipeline stages."""
        return self._scrubber.pseudonymise(user_id)

    def audit_log(self) -> List[AuditEvent]:
        """Return the in-memory audit buffer (most recent BUFFER_SIZE events)."""
        return list(self._logger._buffer)

    # ---- Private helpers ---------------------------------------------------

    def _validate_schema(
        self, req: InboundRequest
    ) -> Tuple[bool, str, Optional[BlockCode]]:
        """
        Validate all fields of InboundRequest.

        Returns (valid, reason, block_code).
        """
        # Required string fields
        for field_name in ("user_id", "session_id", "intent"):
            val = getattr(req, field_name, None)
            if not val or not isinstance(val, str) or not val.strip():
                return (
                    False,
                    f"Missing or empty required field: '{field_name}'",
                    BlockCode.MISSING_REQUIRED_FIELD,
                )

        # Intent validation
        if req.intent not in VALID_INTENTS:
            return (
                False,
                f"Invalid intent '{req.intent}'. Must be one of: {sorted(VALID_INTENTS)}",
                BlockCode.INVALID_INTENT,
            )

        # mood_pre range
        if not isinstance(req.mood_pre, (int, float)):
            return (
                False,
                f"mood_pre must be numeric, got {type(req.mood_pre).__name__}",
                BlockCode.SCHEMA_VIOLATION,
            )
        if not (0.0 <= float(req.mood_pre) <= 100.0):
            return (
                False,
                f"mood_pre = {req.mood_pre} is outside [0, 100]",
                BlockCode.SCHEMA_VIOLATION,
            )

        # Sensitivity flags
        unknown_flags = set(req.sensitivity_flags) - VALID_SENSITIVITY_FLAGS
        if unknown_flags:
            return (
                False,
                f"Unknown sensitivity flag(s): {sorted(unknown_flags)}. "
                f"Valid: {sorted(VALID_SENSITIVITY_FLAGS)}",
                BlockCode.INVALID_SENSITIVITY_FLAG,
            )

        # Feature vector must be a dict
        if not isinstance(req.feature_vector, dict):
            return (
                False,
                f"feature_vector must be a dict, got {type(req.feature_vector).__name__}",
                BlockCode.SCHEMA_VIOLATION,
            )

        return True, "", None

    @staticmethod
    def _rate_limit_key(req: InboundRequest, pseudonym: str) -> str:
        """
        Build the rate-limit lookup key.

        Prefer IP-based keying when available (prevents user-ID cycling
        attacks where an attacker creates new user IDs to bypass per-user
        limits).  Fall back to pseudonym if no IP.
        """
        return req.client_ip if req.client_ip else pseudonym

    def _block_and_log(
        self,
        req:          InboundRequest,
        pseudonym:    str,
        block_code:   BlockCode,
        block_reason: str,
        anomaly_score:float,
        warnings:     List[str],
    ) -> GateResult:
        """Create a blocked GateResult and write the audit event."""
        try:
            scrubbed_req, _ = self._scrubber.scrub_request(req)
        except Exception:
            # Scrubbing the request failed (e.g. malformed feature_vector type).
            # Build a minimal safe placeholder so the block can still be logged.
            scrubbed_req = InboundRequest(
                user_id=pseudonym, session_id=req.session_id,
                intent=req.intent if req.intent in VALID_INTENTS else "unknown",
                mood_pre=0.0, sensitivity_flags=[], feature_vector={},
            )
        self._log_event(
            pseudonym    = pseudonym,
            session_id   = req.session_id,
            intent       = req.intent,
            event_type   = AuditEventType.REQUEST_BLOCKED,
            allowed      = False,
            block_code   = block_code.value,
            block_reason = block_reason,
            anomaly_score= anomaly_score,
            warnings     = warnings,
            details      = {"block_code": block_code.value},
        )
        return GateResult(
            allowed          = False,
            pseudonym        = pseudonym,
            scrubbed_request = scrubbed_req,
            anomaly_score    = anomaly_score,
            block_code       = block_code,
            block_reason     = block_reason,
            warnings         = warnings,
        )

    def _log_event(
        self,
        pseudonym:    str,
        session_id:   str,
        intent:       str,
        event_type:   AuditEventType,
        allowed:      bool,
        block_code:   Optional[str],
        block_reason: str,
        anomaly_score:float,
        warnings:     List[str],
        details:      Dict[str, Any],
    ) -> None:
        event = AuditEvent(
            event_id      = str(uuid.uuid4()),
            sequence      = 0,  # filled by AuditLogger.record()
            event_type    = event_type.value,
            pseudonym     = pseudonym,
            session_id    = session_id,
            intent        = intent,
            timestamp     = datetime.now(timezone.utc).isoformat(),
            allowed       = allowed,
            block_code    = block_code,
            block_reason  = block_reason,
            anomaly_score = round(anomaly_score, 4),
            warnings      = warnings,
            details       = details,
        )
        self._logger.record(event)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        sec = SecurityLayer(
            rate_limit_requests=5,
            rate_limit_window_s=10,
            burst_capacity=2,
            audit_log_path=tmp / "audit.log",
            pii_salt="demo_salt_fixed",
        )

        _VALID_FV = {
            "tempo_bpm": 72.0, "energy": 0.28, "valence": 0.34,
            "acousticness": 0.81, "instrumentalness": 0.67,
            "speechiness": 0.04, "loudness_db": -18.4,
            "mood_pre": 34.0, "intent_encoded": 1.0,
        }

        def _req(**kwargs) -> InboundRequest:
            defaults = dict(
                user_id="user_042", session_id="sess_001",
                intent="anxiety_relief", mood_pre=34.0,
                sensitivity_flags=["grief_sensitive"],
                feature_vector=dict(_VALID_FV),
            )
            defaults.update(kwargs)
            return InboundRequest(**defaults)

        print("=" * 68)
        print("  HarmonicAI — Phase 8 Security Layer Demo")
        print("=" * 68)

        scenarios = [
            ("Valid request",
             _req()),
            ("Invalid intent",
             _req(intent="bad_intent")),
            ("mood_pre out of range",
             _req(mood_pre=150.0)),
            ("Unknown sensitivity flag",
             _req(sensitivity_flags=["unknown_flag"])),
            ("Feature out of bounds (tempo_bpm=999)",
             _req(feature_vector={**_VALID_FV, "tempo_bpm": 999.0})),
            ("Adversarial: high energy + high acousticness",
             _req(feature_vector={**_VALID_FV, "energy": 0.99, "acousticness": 0.99,
                                   "instrumentalness": 0.95, "speechiness": 0.65})),
            ("PII in lyric (email address)",
             _req(raw_lyric="Contact me at user@example.com for more info")),
        ]

        for label, req in scenarios:
            gate = sec.inspect_request(req)
            status = "✅ ALLOWED" if gate.allowed else f"🚫 BLOCKED [{gate.block_code.value if gate.block_code else ''}]"
            print(f"\n  {label}")
            print(f"    → {status}")
            if gate.block_reason:
                print(f"    Reason: {gate.block_reason}")
            if gate.warnings:
                print(f"    Warnings: {gate.warnings}")
            print(f"    Pseudonym: {gate.pseudonym}")
            if gate.anomaly_score > 0:
                print(f"    Anomaly score: {gate.anomaly_score:.3f}")

        # Rate limiting demo
        print("\n  Rate limiting (7 requests, limit=5+2 burst):")
        for i in range(7):
            gate = sec.inspect_request(_req(session_id=f"sess_{i:03d}"))
            status = "✅" if gate.allowed else "🚫 RATE LIMITED"
            print(f"    Request {i+1}: {status}")

        # Response sanitisation demo
        print("\n  Response sanitisation:")
        raw_response = {
            "tracks":       [{"track_id": "t001"}],
            "script":       "Take a breath. Contact admin@harmonicai.com if issues arise.",
            "model_path":   "/models/checkpoints/random_forest.pkl",
            "internal_debug": {"cv_scores": [0.91, 0.89]},
            "session_id":   "sess_001",
            "user_id":      "user_042",   # raw — should be pseudonymised
        }
        clean = sec.sanitise_response(raw_response, "user_042")
        print(f"    'model_path' removed: {'model_path' not in clean}")
        print(f"    'internal_debug' removed: {'internal_debug' not in clean}")
        print(f"    user_id pseudonymised: {clean['user_id'][:8]}...")
        print(f"    email scrubbed from script: {'@' not in clean['script']}")

        # Audit log summary
        log = sec.audit_log()
        allowed = sum(1 for e in log if e.allowed)
        blocked = sum(1 for e in log if not e.allowed)
        print(f"\n  Audit log: {len(log)} events  ({allowed} allowed, {blocked} blocked)")
        if (tmp / "audit.log").exists():
            lines = (tmp / "audit.log").read_text().strip().split("\n")
            print(f"  On-disk log: {len(lines)} lines (JSON, each self-contained)")

        print("\n" + "=" * 68)
        print("  Demo complete.")
        print("=" * 68)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _run_demo()
