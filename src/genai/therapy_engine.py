"""
src/genai/therapy_engine.py
===========================
Phase 6 — Therapeutic Script Generation Engine

Purpose
-------
Takes the top-5 track recommendations from Phase 3 (mood_classifier.py) and
the user's current session context, then generates a personalized, clinically-
grounded mindfulness/breathing script via an LLM API.

Design principles
-----------------
1. PROMPT CONDITIONING   — every prompt is assembled from intent-specific
                           templates, user mood state, and lead-track properties.
                           The LLM never receives a generic "write something
                           calming" instruction.

2. TEMPERATURE CONTROL   — temperature is not a global constant. Each session
                           intent has its own calibrated value.  Sleep induction
                           demands consistency (low temperature); mood uplift
                           tolerates creativity (higher temperature).

3. THERAPEUTIC GUARDRAILS — generated text is post-processed through a rule
                             set before being returned.  Any flagged output is
                             replaced with a vetted fallback script rather than
                             passed to the user.

4. FAIL LOUD             — consistent with loader.py: schema violations and
                           API errors raise immediately with a descriptive
                           message; they are never silently swallowed.

5. DETERMINISTIC MOCK    — the engine ships with a DeterministicMock backend
                           so that all other pipeline stages can run without
                           an API key. The real backend (AnthropicBackend) is
                           a drop-in replacement.

Session intent → temperature mapping (rationale)
-------------------------------------------------
sleep_induction : 0.30  repetition and predictability are therapeutic
anxiety_relief  : 0.40  grounding exercises need reliable structure
grief_processing: 0.55  warmth matters; some variation is human-feeling
deep_focus      : 0.50  precision over creativity
mood_uplift     : 0.70  energising variety is part of the intervention

Usage
-----
    from src.genai.therapy_engine import TherapyEngine, SessionContext

    engine = TherapyEngine()                          # real API
    # engine = TherapyEngine(backend="mock")          # no API key needed

    ctx = SessionContext(
        user_id          = "user_042",
        intent           = "anxiety_relief",
        mood_pre         = 34,
        sensitivity_flags= ["grief_sensitive"],
        top_tracks       = tracks_df.head(5),         # DataFrame from Phase 3
    )

    result = engine.generate(ctx)
    print(result.script)
    print(result.meta)
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_INTENTS = {
    "sleep_induction",
    "anxiety_relief",
    "grief_processing",
    "deep_focus",
    "mood_uplift",
}

# Intent → LLM temperature (see module docstring for rationale)
INTENT_TEMPERATURE: Dict[str, float] = {
    "sleep_induction":  0.30,
    "anxiety_relief":   0.40,
    "grief_processing": 0.55,
    "deep_focus":       0.50,
    "mood_uplift":      0.70,
}

# Intent → target script length in words (kept short; therapeutic, not literary)
INTENT_MAX_WORDS: Dict[str, int] = {
    "sleep_induction":  200,
    "anxiety_relief":   160,
    "grief_processing": 180,
    "deep_focus":       140,
    "mood_uplift":      150,
}

# Model to call
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    """
    Everything the therapy engine needs about the current session.

    Parameters
    ----------
    user_id : str
        Opaque user identifier. Used only for logging — no PII in prompts.
    intent : str
        One of VALID_INTENTS. Drives template selection and temperature.
    mood_pre : float
        User's self-reported mood on entry (0–100 scale, lower = worse).
    top_tracks : pd.DataFrame
        The top-5 ranked tracks from mood_classifier.py. Must have columns:
        track_id, tempo_bpm, energy, valence, acousticness,
        instrumentalness, cluster_name.
    sensitivity_flags : list[str]
        User's active sensitivity flags from users_enriched.csv.
        Relevant values: "grief_sensitive", "anxiety_prone", "sleep_disorder".
    session_id : str, optional
        Used to correlate logs with feedback events in Phase 7.
    """
    user_id:           str
    intent:            str
    mood_pre:          float
    top_tracks:        pd.DataFrame
    sensitivity_flags: List[str]  = field(default_factory=list)
    session_id:        str        = ""

    def __post_init__(self) -> None:
        if self.intent not in VALID_INTENTS:
            raise ValueError(
                f"Unknown intent '{self.intent}'. "
                f"Must be one of: {sorted(VALID_INTENTS)}"
            )
        if not (0.0 <= self.mood_pre <= 100.0):
            raise ValueError(
                f"mood_pre must be in [0, 100], got {self.mood_pre}"
            )
        required_cols = {
            "track_id", "tempo_bpm", "energy", "valence",
            "acousticness", "instrumentalness", "cluster_name",
        }
        missing = required_cols - set(self.top_tracks.columns)
        if missing:
            raise ValueError(
                f"top_tracks DataFrame is missing columns: {sorted(missing)}"
            )
        if len(self.top_tracks) == 0:
            raise ValueError("top_tracks must contain at least one track.")


@dataclass
class TherapyScript:
    """
    Structured output from TherapyEngine.generate().

    Attributes
    ----------
    script : str
        The generated (and guardrail-validated) mindfulness script.
    intent : str
        Echo of the originating intent.
    lead_track_id : str
        track_id of the track that anchored the prompt (rank 1 recommendation).
    temperature_used : float
        Actual temperature sent to the LLM.
    word_count : int
        Word count of the returned script.
    guardrail_triggered : bool
        True if a guardrail fired and the fallback script was substituted.
    guardrail_reason : str
        Human-readable explanation if guardrail_triggered is True.
    backend : str
        Which backend produced this script ("anthropic" or "mock").
    latency_ms : int
        Milliseconds from API call start to response received.
    meta : dict
        All of the above packaged as a dict for logging / Phase 7 feedback.
    """
    script:               str
    intent:               str
    lead_track_id:        str
    temperature_used:     float
    word_count:           int
    guardrail_triggered:  bool = False
    guardrail_reason:     str  = ""
    backend:              str  = "anthropic"
    latency_ms:           int  = 0

    @property
    def meta(self) -> Dict[str, Any]:
        return {
            "intent":              self.intent,
            "lead_track_id":       self.lead_track_id,
            "temperature_used":    self.temperature_used,
            "word_count":          self.word_count,
            "guardrail_triggered": self.guardrail_triggered,
            "guardrail_reason":    self.guardrail_reason,
            "backend":             self.backend,
            "latency_ms":          self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class TherapyPromptBuilder:
    """
    Constructs fully-conditioned system + user prompts for each session intent.

    Why separate from TherapyEngine?
    ---------------------------------
    Prompt engineering is a distinct concern from API orchestration.  Keeping
    the builder isolated means prompt templates can be tested, versioned, and
    swapped without touching API or guardrail logic.
    """

    # ---- Intent-specific opening instructions ----------------------------
    _INTENT_INSTRUCTIONS: Dict[str, str] = {
        "sleep_induction": (
            "Guide the listener through a slow, progressive body-scan relaxation. "
            "Use long, unhurried sentences. Lean heavily on sleep-safe imagery: "
            "still water, drifting clouds, deepening darkness. "
            "Avoid any instruction that requires sustained attention or mental effort."
        ),
        "anxiety_relief": (
            "Lead a structured grounding exercise using the 5-4-3-2-1 sensory "
            "technique or box breathing. Anchor every instruction to the present "
            "moment. Use short, confident sentences. Avoid future-oriented language "
            "('you will', 'soon you'll feel') — stay fully in the now."
        ),
        "grief_processing": (
            "Hold space for the listener's emotions without rushing toward resolution. "
            "Validate rather than redirect. Use gentle, inclusive language: "
            "'if you want to', 'you might notice'. "
            "Never tell the listener how to feel or impose a timeline for healing."
        ),
        "deep_focus": (
            "Prepare the listener for a sustained concentration session. "
            "Use crisp, directive language. Guide a brief breath-reset (3–4 cycles), "
            "then pivot to mental clearing — labelling and releasing distractions. "
            "End with a clear, energising cue to begin work."
        ),
        "mood_uplift": (
            "Open with a warm acknowledgement of the listener's effort to feel better. "
            "Guide a brief gratitude or positive-attention exercise. "
            "Use varied, expressive language. Finish with an affirming, "
            "forward-looking close — something the listener can carry into their day."
        ),
    }

    # ---- Sensitivity-flag modifiers ---------------------------------------
    _SENSITIVITY_MODIFIERS: Dict[str, str] = {
        "grief_sensitive": (
            "This user has flagged grief sensitivity. "
            "Avoid all references to loss, death, endings, or absence. "
            "Do not use the words 'gone', 'lost', 'alone', or 'empty'."
        ),
        "anxiety_prone": (
            "This user is anxiety-prone. "
            "Do not use urgency language ('quickly', 'try not to', 'don't let'). "
            "Avoid catastrophising framing even as a contrast ('instead of panicking...')."
        ),
        "sleep_disorder": (
            "This user has a sleep disorder. "
            "Never reference difficulty falling asleep or wakefulness. "
            "Frame all sleep guidance as natural and effortless."
        ),
    }

    # ---- Mood-state descriptors (mood_pre is 0–100, lower = worse) -------
    @staticmethod
    def _mood_descriptor(mood_pre: float) -> str:
        if mood_pre <= 20:
            return "very low — the listener may be in significant distress"
        elif mood_pre <= 40:
            return "below average — the listener is struggling"
        elif mood_pre <= 60:
            return "moderate — the listener is seeking support"
        elif mood_pre <= 80:
            return "good — the listener wants to maintain their state"
        else:
            return "high — the listener is in an excellent baseline state"

    # ---- Track-property descriptors ---------------------------------------
    @staticmethod
    def _track_descriptor(track: pd.Series) -> str:
        tempo   = float(track["tempo_bpm"])
        energy  = float(track["energy"])
        valence = float(track["valence"])
        cluster = str(track["cluster_name"])

        tempo_label   = "slow" if tempo < 80 else ("moderate" if tempo < 120 else "fast")
        energy_label  = "gentle" if energy < 0.4 else ("moderate" if energy < 0.7 else "energetic")
        valence_label = "melancholic" if valence < 0.35 else ("neutral" if valence < 0.65 else "uplifting")

        return (
            f"{tempo_label} tempo ({tempo:.0f} BPM), "
            f"{energy_label} energy, "
            f"{valence_label} tone, "
            f"acoustic cluster: {cluster}"
        )

    def build(self, ctx: SessionContext) -> Dict[str, str]:
        """
        Return {"system": ..., "user": ...} prompt dict for the LLM.

        The system prompt establishes the therapeutic persona and hard rules.
        The user prompt supplies all session-specific conditioning signals.
        """
        lead_track  = ctx.top_tracks.iloc[0]
        track_desc  = self._track_descriptor(lead_track)
        mood_desc   = self._mood_descriptor(ctx.mood_pre)
        intent_inst = self._INTENT_INSTRUCTIONS[ctx.intent]
        max_words   = INTENT_MAX_WORDS[ctx.intent]

        # Build sensitivity modifier block
        sensitivity_block = ""
        for flag in ctx.sensitivity_flags:
            if flag in self._SENSITIVITY_MODIFIERS:
                sensitivity_block += f"\n- {self._SENSITIVITY_MODIFIERS[flag]}"
        if sensitivity_block:
            sensitivity_block = "\n\nSENSITIVITY CONSTRAINTS:" + sensitivity_block

        system_prompt = textwrap.dedent(f"""
            You are a licensed mindfulness and therapeutic music guide with
            training in evidence-based practices: MBSR, CBT-informed
            relaxation, and music therapy.  You write short, spoken-word
            scripts that a user hears at the start of a therapeutic music
            session.

            ABSOLUTE RULES (never violate):
            1. Never offer a medical diagnosis, medication advice, or clinical
               treatment recommendations.
            2. Never dismiss, minimise, or fix the listener's emotions.
            3. Never use alarming, shaming, or directive language
               ("you must", "you should", "stop feeling").
            4. Write for a speaking pace of ~130 words per minute.
               Maximum script length: {max_words} words.
            5. Do not include a title, section headers, or speaker labels.
               Return only the script text itself.
            6. Write in second person ("you", "your"), present tense.
        """).strip()

        user_prompt = textwrap.dedent(f"""
            SESSION CONTEXT
            ---------------
            Intent          : {ctx.intent}
            Listener mood   : {mood_desc} (score {ctx.mood_pre:.0f}/100)
            Lead track      : {track_desc}
            {sensitivity_block}

            TASK
            ----
            {intent_inst}

            Compose a {max_words}-word-maximum spoken mindfulness script
            that is fully conditioned on the session context above.
            The script should feel like a natural opening for music with
            the acoustic properties described — it should complement,
            not contradict, the sonic character of the track.
        """).strip()

        return {"system": system_prompt, "user": user_prompt}


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

class TherapeuticGuardrails:
    """
    Post-generation validation layer.

    Each check is a named method returning (passed: bool, reason: str).
    All checks run even if an earlier one fails — the complete reason
    set is logged for human review.

    Why post-generation rather than constrained decoding?
    ------------------------------------------------------
    Constrained decoding (e.g., grammar-guided sampling) requires model-
    level access.  Post-generation checks are model-agnostic and can be
    updated without touching the LLM pipeline.  The trade-off is that
    a failed check discards a complete generation.  Given the short script
    length (~150 words), regeneration cost is low.
    """

    # Terms that must never appear in a therapeutic script
    _HARD_BLOCK_TERMS = [
        # clinical / diagnostic
        r"\bdiagnos\w*\b", r"\bdisorder\b", r"\bmedication\b", r"\bprescri\w+\b",
        r"\bsymptom\w*\b", r"\btreatment\b",
        # catastrophising / minimising
        r"\bdon't worry\b", r"\bjust relax\b", r"\bstop feeling\b",
        r"\bget over\b", r"\bsnap out\b",
        # self-harm adjacent
        r"\bhurt yourself\b", r"\bno point\b", r"\bgive up\b",
        r"\bend it\b", r"\bworthless\b",
    ]

    # Script must contain at least one of these engagement signals
    _ENGAGEMENT_SIGNALS = [
        r"\bbreathe?\b", r"\bnotice\b", r"\bfeel\b", r"\bbody\b",
        r"\bpresent\b", r"\bmoment\b", r"\bground\w*\b", r"\bsense\b",
        r"\brelax\b", r"\bgentle\b", r"\bsoft\b", r"\bwarm\b",
    ]

    def validate(self, script: str) -> tuple[bool, str]:
        """
        Run all guardrail checks.

        Returns
        -------
        (passed, reason)
            passed=True means the script is safe to return.
            reason is empty string when passed=True.
        """
        reasons: List[str] = []

        ok, r = self._check_hard_block_terms(script)
        if not ok:
            reasons.append(r)

        ok, r = self._check_engagement_signals(script)
        if not ok:
            reasons.append(r)

        ok, r = self._check_length(script)
        if not ok:
            reasons.append(r)

        ok, r = self._check_not_empty(script)
        if not ok:
            reasons.append(r)

        passed = len(reasons) == 0
        return passed, "; ".join(reasons)

    def _check_hard_block_terms(self, script: str) -> tuple[bool, str]:
        lower = script.lower()
        hits  = [
            pattern for pattern in self._HARD_BLOCK_TERMS
            if re.search(pattern, lower)
        ]
        if hits:
            return False, f"Hard-blocked terms found: {hits}"
        return True, ""

    def _check_engagement_signals(self, script: str) -> tuple[bool, str]:
        lower = script.lower()
        found = any(re.search(p, lower) for p in self._ENGAGEMENT_SIGNALS)
        if not found:
            return (
                False,
                "Script lacks therapeutic engagement signals "
                "(breath, body, notice, ground, etc.)"
            )
        return True, ""

    def _check_length(self, script: str) -> tuple[bool, str]:
        words = len(script.split())
        if words < 30:
            return False, f"Script too short ({words} words, minimum 30)"
        if words > 350:
            return False, f"Script too long ({words} words, maximum 350)"
        return True, ""

    def _check_not_empty(self, script: str) -> tuple[bool, str]:
        if not script.strip():
            return False, "Script is empty"
        return True, ""


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class _BaseBackend:
    """Abstract base — concrete backends implement `complete()`."""

    name: str = "base"

    def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   float,
        max_tokens:    int,
    ) -> tuple[str, int]:
        """
        Returns (text, latency_ms).
        Raises RuntimeError on any API failure.
        """
        raise NotImplementedError


class AnthropicBackend(_BaseBackend):
    """
    Calls the Anthropic Messages API.

    Requires:
        pip install anthropic

    The API key is read from the ANTHROPIC_API_KEY environment variable
    (standard Anthropic SDK behaviour — no key is ever hard-coded).
    """

    name = "anthropic"

    def __init__(self) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "AnthropicBackend requires the anthropic package. "
                "Install it with: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   float,
        max_tokens:    int,
    ) -> tuple[str, int]:
        import anthropic

        t0 = time.monotonic()
        try:
            message = self._client.messages.create(
                model      = ANTHROPIC_MODEL,
                max_tokens = max_tokens,
                temperature= temperature,
                system     = system_prompt,
                messages   = [{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise RuntimeError(f"Anthropic API call failed: {exc}") from exc

        latency_ms = int((time.monotonic() - t0) * 1000)
        text = message.content[0].text.strip()
        return text, latency_ms


class DeterministicMock(_BaseBackend):
    """
    Returns a pre-written fallback script for each intent.

    Used in:
    - Unit tests
    - CI pipelines without API keys
    - Demo mode
    - Phase 7 drift detection tests (no LLM calls needed)

    The mock is deterministic: same intent always returns same script,
    which makes regression testing reliable.
    """

    name = "mock"

    _MOCK_SCRIPTS: Dict[str, str] = {
        "sleep_induction": (
            "Find a comfortable position and allow your eyes to close. "
            "Take a slow breath in through your nose, and release it gently "
            "through your mouth. Notice the weight of your body settling. "
            "With each breath, you feel yourself sinking a little further into "
            "stillness. Let the music carry you — there is nothing to do right "
            "now, nowhere to be. Your body knows how to rest. Simply follow "
            "the rhythm of your breath, as soft and unhurried as the tide "
            "drawing back from the shore. You are safe. You are held. "
            "Allow yourself to drift."
        ),
        "anxiety_relief": (
            "Let's ground you right here, right now. "
            "Bring your attention to your feet — feel the surface beneath them. "
            "Take a breath in for four counts: one, two, three, four. "
            "Hold for four: one, two, three, four. "
            "Release for four: one, two, three, four. "
            "Good. Now notice five things you can see. Four things you can "
            "physically feel. Three sounds around you. Two things you can smell. "
            "One thing you can taste. You are here. You are steady. "
            "The present moment is available to you right now, and that is enough."
        ),
        "grief_processing": (
            "You don't have to be okay right now. "
            "If you want to, bring one hand to your chest and feel your heartbeat — "
            "steady, present, yours. Notice whatever is here. "
            "You don't need to name it or change it. "
            "Let the music hold some of the weight alongside you. "
            "Grief moves in its own time, and there is no way through it "
            "that is wrong. You might notice your breath changing — "
            "let it. Your body is wise. You are allowed to feel exactly "
            "what you feel."
        ),
        "deep_focus": (
            "Let's clear the space for your work. "
            "Take three full breaths — slow in, full out. "
            "Notice any thoughts waiting for your attention and let them "
            "know they can wait a little longer. "
            "You have everything you need to begin. "
            "Your focus is a resource you can direct deliberately. "
            "Set one clear intention for this session. "
            "When you're ready, let the music mark the start. "
            "You are here. You are capable. Begin."
        ),
        "mood_uplift": (
            "Take a moment to acknowledge that you showed up today — "
            "that matters. "
            "Bring your attention to something small that's working: "
            "your breath moving, the sensation of being alive in this moment. "
            "Notice one thing in your field of awareness that you appreciate — "
            "however small. Let that recognition settle in your chest. "
            "You are more resilient than yesterday's hard moments. "
            "Let the music meet you where you are and carry you forward. "
            "Something good is available to you today."
        ),
    }

    def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   float,
        max_tokens:    int,
    ) -> tuple[str, int]:
        # Parse intent from user_prompt (it's always explicitly labelled)
        intent_match = re.search(r"Intent\s*:\s*(\w+)", user_prompt)
        intent = intent_match.group(1) if intent_match else "anxiety_relief"

        script = self._MOCK_SCRIPTS.get(
            intent,
            self._MOCK_SCRIPTS["anxiety_relief"]  # safe default
        )
        return script, 0  # 0ms — no real network call


# ---------------------------------------------------------------------------
# Fallback scripts (used when guardrails fire on real LLM output)
# ---------------------------------------------------------------------------

# Identical content as mock — these are human-vetted, always safe.
_FALLBACK_SCRIPTS = DeterministicMock._MOCK_SCRIPTS


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class TherapyEngine:
    """
    Orchestrates prompt building, LLM calls, and guardrail validation.

    Parameters
    ----------
    backend : str
        "anthropic"  — real API (requires ANTHROPIC_API_KEY env var)
        "mock"       — deterministic mock, no API key required

    max_retries : int
        Number of times to retry a guardrail-failed generation before
        substituting the fallback script.  Retries use a slightly higher
        temperature to encourage different output.

    Example
    -------
    >>> engine = TherapyEngine(backend="mock")
    >>> ctx = SessionContext(
    ...     user_id="u001", intent="anxiety_relief", mood_pre=34.0,
    ...     top_tracks=tracks_df.head(5), sensitivity_flags=[]
    ... )
    >>> result = engine.generate(ctx)
    >>> print(result.script)
    """

    def __init__(
        self,
        backend:     str = "anthropic",
        max_retries: int = 2,
    ) -> None:
        if backend == "anthropic":
            self._backend: _BaseBackend = AnthropicBackend()
        elif backend == "mock":
            self._backend = DeterministicMock()
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'anthropic' or 'mock'."
            )

        self._prompt_builder = TherapyPromptBuilder()
        self._guardrails     = TherapeuticGuardrails()
        self._max_retries    = max_retries

    def generate(self, ctx: SessionContext) -> TherapyScript:
        """
        Generate a therapeutic script for the given session context.

        Flow
        ----
        1. Build conditioned system + user prompts.
        2. Determine temperature from intent.
        3. Call LLM backend (with retry on guardrail failure).
        4. Validate output through guardrails.
        5. Return TherapyScript (or substitute fallback if retries exhausted).
        """
        prompts      = self._prompt_builder.build(ctx)
        temperature  = INTENT_TEMPERATURE[ctx.intent]
        max_words    = INTENT_MAX_WORDS[ctx.intent]
        max_tokens   = int(max_words * 1.8)   # ~1.8 tokens per word, generous headroom
        lead_track   = ctx.top_tracks.iloc[0]

        script        = ""
        latency_ms    = 0
        guardrail_ok  = False
        fail_reason   = ""

        for attempt in range(self._max_retries + 1):
            # Slightly raise temperature on retry to escape a stuck generation
            effective_temp = min(temperature + attempt * 0.08, 1.0)

            raw_script, call_latency = self._backend.complete(
                system_prompt = prompts["system"],
                user_prompt   = prompts["user"],
                temperature   = effective_temp,
                max_tokens    = max_tokens,
            )
            latency_ms += call_latency

            guardrail_ok, fail_reason = self._guardrails.validate(raw_script)

            if guardrail_ok:
                script = raw_script
                break

            # Guardrail fired — log and retry
            print(
                f"[TherapyEngine] Attempt {attempt + 1}/{self._max_retries + 1} "
                f"guardrail triggered for intent='{ctx.intent}': {fail_reason}"
            )

        guardrail_triggered = not guardrail_ok
        if guardrail_triggered:
            # All retries exhausted — substitute human-vetted fallback
            script = _FALLBACK_SCRIPTS.get(
                ctx.intent,
                _FALLBACK_SCRIPTS["anxiety_relief"]
            )
            print(
                f"[TherapyEngine] Fallback substituted for "
                f"user='{ctx.user_id}' intent='{ctx.intent}'"
            )

        return TherapyScript(
            script              = script,
            intent              = ctx.intent,
            lead_track_id       = str(lead_track["track_id"]),
            temperature_used    = temperature,
            word_count          = len(script.split()),
            guardrail_triggered = guardrail_triggered,
            guardrail_reason    = fail_reason if guardrail_triggered else "",
            backend             = self._backend.name,
            latency_ms          = latency_ms,
        )

    def batch_generate(
        self, contexts: List[SessionContext]
    ) -> List[TherapyScript]:
        """
        Generate scripts for multiple sessions sequentially.

        Note: Not parallelised — LLM APIs have rate limits and
        therapeutic scripts should never be rushed.
        """
        return [self.generate(ctx) for ctx in contexts]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _make_demo_tracks() -> pd.DataFrame:
    """Create a minimal tracks DataFrame for demo / testing."""
    return pd.DataFrame([
        {
            "track_id":        "track_0001",
            "tempo_bpm":       72.0,
            "energy":          0.28,
            "valence":         0.34,
            "acousticness":    0.81,
            "instrumentalness":0.67,
            "speechiness":     0.04,
            "loudness_db":    -18.4,
            "cluster_name":   "low_arousal_acoustic",
        },
        {
            "track_id":        "track_0002",
            "tempo_bpm":       65.0,
            "energy":          0.22,
            "valence":         0.28,
            "acousticness":    0.89,
            "instrumentalness":0.74,
            "speechiness":     0.03,
            "loudness_db":    -21.1,
            "cluster_name":   "minimal_instrumental",
        },
    ])


def _run_demo() -> None:
    """
    Demonstrate the full Phase 6 pipeline using the mock backend.
    Prints one script per intent so the output can be visually inspected.
    """
    engine = TherapyEngine(backend="mock")
    demo_tracks = _make_demo_tracks()

    print("=" * 72)
    print("HarmonicAI — Phase 6 Therapy Engine Demo (mock backend)")
    print("=" * 72)

    demo_sessions = [
        SessionContext(
            user_id           = "user_demo_01",
            intent            = "anxiety_relief",
            mood_pre          = 31.0,
            top_tracks        = demo_tracks,
            sensitivity_flags = ["grief_sensitive"],
            session_id        = "sess_demo_001",
        ),
        SessionContext(
            user_id           = "user_demo_02",
            intent            = "sleep_induction",
            mood_pre          = 44.0,
            top_tracks        = demo_tracks,
            sensitivity_flags = ["sleep_disorder"],
            session_id        = "sess_demo_002",
        ),
        SessionContext(
            user_id           = "user_demo_03",
            intent            = "grief_processing",
            mood_pre          = 22.0,
            top_tracks        = demo_tracks,
            sensitivity_flags = ["grief_sensitive", "anxiety_prone"],
            session_id        = "sess_demo_003",
        ),
        SessionContext(
            user_id           = "user_demo_04",
            intent            = "mood_uplift",
            mood_pre          = 55.0,
            top_tracks        = demo_tracks,
            sensitivity_flags = [],
            session_id        = "sess_demo_004",
        ),
        SessionContext(
            user_id           = "user_demo_05",
            intent            = "deep_focus",
            mood_pre          = 68.0,
            top_tracks        = demo_tracks,
            sensitivity_flags = [],
            session_id        = "sess_demo_005",
        ),
    ]

    for ctx in demo_sessions:
        result = engine.generate(ctx)
        print(f"\n{'─' * 72}")
        print(f"  Intent            : {result.intent}")
        print(f"  Lead track        : {result.lead_track_id}")
        print(f"  Temperature used  : {result.temperature_used}")
        print(f"  Word count        : {result.word_count}")
        print(f"  Backend           : {result.backend}")
        print(f"  Guardrail fired   : {result.guardrail_triggered}")
        print(f"{'─' * 72}")
        print()
        # Wrap at 72 chars for terminal readability
        for line in textwrap.wrap(result.script, width=72):
            print(f"  {line}")
        print()

    print("=" * 72)
    print("Demo complete. meta dict from last result:")
    print(json.dumps(result.meta, indent=2))
    print("=" * 72)


if __name__ == "__main__":
    _run_demo()
