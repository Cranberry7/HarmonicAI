import argparse
import pickle
import sys
import textwrap
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.feedback.drift_detector import INTENT_ENCODING
from src.genai.therapy_engine import TherapyEngine, SessionContext
from src.security.security_layer import SecurityLayer, InboundRequest

# ── Colour helpers ─────────────────────────────────────────────────────────

_NO_COLOUR = False
def _c(text: str, code: str) -> str:
    if _NO_COLOUR: return text
    return f"\033[{code}m{text}\033[0m"

def _green(t):  return _c(t, "32")
def _cyan(t):   return _c(t, "36")
def _yellow(t): return _c(t, "33")
def _bold(t):   return _c(t, "1")
def _dim(t):    return _c(t, "2")
def _red(t):    return _c(t, "31")

def _header(title: str):
    bar = "─" * 68
    print(f"\n{_cyan(bar)}")
    print(f"  {_bold(title)}")
    print(_cyan(bar))

def _ok(msg: str):    print(f"  {_green('✓')}  {msg}")
def _info(msg: str):  print(f"  {_dim('·')}  {msg}")
def _warn(msg: str):  print(f"  {_yellow('⚠')}  {msg}")
def _block(msg: str): print(f"  {_red('✗')}  {msg}")

def run_interactive_request(
    tracks_df:       pd.DataFrame,
    rf_pipeline:     object,
    feature_list:    list[str],
    filter_artefact: dict,
    therapy_engine:  TherapyEngine,
    security:        SecurityLayer,
    user_mood:       float = 31.0,
    user_intent:     str = "anxiety_relief",
):
    _header("LIVE REQUEST LIFECYCLE  ·  Interactive Session")

    USER_ID           = "user_0042"
    SESSION_ID        = "sess_live_001"
    INTENT            = user_intent
    MOOD_PRE          = float(user_mood)
    SENSITIVITY_FLAGS = ["grief_sensitive"]

    print(f"\n  {_bold('Incoming user request:')}")
    _info(f"    intent      = {INTENT}")
    _info(f"    mood_pre    = {MOOD_PRE}/100")
    _info(f"    flags       = {SENSITIVITY_FLAGS}")

    # 1. Security Gate
    print(f"\n  {_bold('Step 1 — Security Gate')}")
    sec_req = InboundRequest(
        user_id           = USER_ID,
        session_id        = SESSION_ID,
        intent            = INTENT,
        mood_pre          = MOOD_PRE,
        sensitivity_flags = SENSITIVITY_FLAGS,
        feature_vector    = {
            "tempo_bpm": 80.0, "energy": 0.3, "valence": 0.4,
            "acousticness": 0.7, "instrumentalness": 0.5,
            "speechiness": 0.04, "loudness_db": -18.0,
            "mood_pre": MOOD_PRE,
            "intent_encoded": float(INTENT_ENCODING[INTENT]),
        },
    )

    gate = security.inspect_request(sec_req)
    if not gate.allowed:
        _block(f"Request BLOCKED: {gate.block_reason}")
        return

    _ok("Security Gate passed.")

    # 2. Acoustic Cluster Pre-Filter
    INTENT_CLUSTER_AFFINITY = {
        "anxiety_relief":   [0, 3, 5],
        "sleep_induction":  [0, 3, 5],
        "grief_processing": [3, 5, 0],
        "mood_uplift":      [1, 2, 0],
        "deep_focus":       [3, 4, 0],
    }
    preferred_clusters = INTENT_CLUSTER_AFFINITY.get(INTENT, list(range(6)))
    candidate_tracks   = tracks_df[tracks_df["cluster_id"].isin(preferred_clusters)].copy()
    
    if len(candidate_tracks) > 200:
        candidate_tracks = candidate_tracks.sample(200, random_state=42)

    # 3. Lyrical Safety Filter
    safety_fn = filter_artefact["filter_fn"]
    
    safe_tracks = []
    for _, row in candidate_tracks.iterrows():
        instr = float(row["instrumentalness"])
        result = safety_fn(
            "soft piano notes drifting" if instr > 0.4 else "gentle waves on a shore",
            instrumentalness=instr,
            sensitivity_flags=SENSITIVITY_FLAGS,
        )
        if result["safe"]:
            safe_tracks.append(row)

    safe_df = pd.DataFrame(safe_tracks)
    _ok(f"Safety filter: {len(safe_df):,} tracks deemed safe.")

    # 4. Mood Classifier
    print(f"\n  {_bold('Recommendations')}")
    safe_df = safe_df.copy()
    safe_df["mood_pre"]       = MOOD_PRE
    safe_df["intent_encoded"] = float(INTENT_ENCODING[INTENT])

    X_score = safe_df[feature_list].fillna(0).values
    probs   = rf_pipeline.predict_proba(X_score)

    classes   = list(rf_pipeline.named_steps["rf"].classes_)
    pos_idx   = classes.index(1) if 1 in classes else 0
    safe_df["p_therapeutic"] = probs[:, pos_idx]

    top5 = safe_df.nlargest(5, "p_therapeutic").reset_index(drop=True)

    print(f"    {'Rank':<5} {'Track ID':<14} {'P(+1)':<8} "
          f"{'Tempo':<8} {'Energy':<8} {'Valence':<8} {'Cluster'}")
    print(f"    {'─'*4:<5} {'─'*13:<14} {'─'*6:<8} "
          f"{'─'*6:<8} {'─'*6:<8} {'─'*6:<8} {'─'*22}")
    for i, row in top5.iterrows():
        print(
            f"    {i+1:<5} {row['track_id']:<14} "
            f"{row['p_therapeutic']:.4f}   "
            f"{row['tempo_bpm']:<8.1f} {row['energy']:<8.3f} "
            f"{row['valence']:<8.3f} {row['cluster_name']}"
        )

    # 5. Therapy Script
    print(f"\n  {_bold('Therapeutic Script')}")
    ctx = SessionContext(
        user_id           = gate.pseudonym,
        session_id        = SESSION_ID,
        intent            = INTENT,
        mood_pre          = MOOD_PRE,
        top_tracks        = top5[[
            "track_id","tempo_bpm","energy","valence",
            "acousticness","instrumentalness","cluster_name",
        ]],
        sensitivity_flags = SENSITIVITY_FLAGS,
    )

    script_result = therapy_engine.generate(ctx)
    if script_result.guardrail_triggered:
        _warn(f"  Guardrail fired → fallback substituted: {script_result.guardrail_reason}")

    print()
    print(_bold("  ┌─ Personalised Opening Script " + "─" * 38 + "┐"))
    for line in textwrap.wrap(script_result.script, width=64):
        print(f"  │  {line:<64}  │")
    print("  └" + "─" * 68 + "┘")
    print()


def main():
    parser = argparse.ArgumentParser(description="HarmonicAI Interactive App")
    parser.add_argument("--mood", type=float, default=None, help="Input your current mood (0-100)")
    parser.add_argument("--intent", type=str, default=None, 
                        choices=["mood_uplift", "sleep_induction", "anxiety_relief", "grief_processing", "deep_focus"], 
                        help="Session intent")
    args = parser.parse_args()

    bar = "═" * 68
    print(f"\n{_cyan(bar)}")
    print(_bold("  HarmonicAI — Interactive Mode"))
    print(_cyan(bar))

    # Check dependencies exist
    tracks_file = ROOT / "data" / "processed" / "tracks_clustered.csv"
    rf_file = ROOT / "models" / "checkpoints" / "random_forest.pkl"
    feat_file = ROOT / "models" / "checkpoints" / "feature_list.pkl"
    safe_file = ROOT / "models" / "checkpoints" / "safety_filter.pkl"

    if not all(f.exists() for f in [tracks_file, rf_file, feat_file, safe_file]):
        print(f"\n{_red('✗ ERROR: Models or data not found.')}")
        print("  Please run `python setup_pipeline.py` first to generate the models.")
        return

    mood_input = args.mood
    if mood_input is None:
        try:
            print("\n  " + _cyan("═" * 66))
            val = input(f"  {_bold('INTERACTIVE PROMPT')} | Enter your current mood (0-100) [default 31.0]: ")
            print("  " + _cyan("═" * 66) + "\n")
            mood_input = float(val) if val.strip() else 31.0
        except ValueError:
            print(f"  {_yellow('⚠')} Invalid input, using default 31.0\n")
            mood_input = 31.0

    intent_input = args.intent
    if intent_input is None:
        intents = ["mood_uplift", "sleep_induction", "anxiety_relief", "grief_processing", "deep_focus"]
        print("  " + _cyan("═" * 66))
        print(f"  {_bold('INTERACTIVE PROMPT')} | Choose your intent:")
        for i, intent in enumerate(intents, 1):
            print(f"    {i}. {intent}")
        val = input(f"  Enter choice (1-5) [default 3]: ")
        print("  " + _cyan("═" * 66) + "\n")
        try:
            choice = int(val) if val.strip() else 3
            if 1 <= choice <= 5:
                intent_input = intents[choice - 1]
            else:
                raise ValueError
        except ValueError:
            print(f"  {_yellow('⚠')} Invalid input, using default anxiety_relief\n")
            intent_input = "anxiety_relief"

    print("  Loading models and data... ", end="", flush=True)

    tracks_df = pd.read_csv(tracks_file)
    with open(rf_file, "rb") as f:
        rf_pipeline = pickle.load(f)
    with open(feat_file, "rb") as f:
        feature_list = pickle.load(f)
    with open(safe_file, "rb") as f:
        filter_dict = pickle.load(f)
        
    # Rebuild filter_fn locally as it was not serialised to prevent pickling issues
    tfidf_pipeline = filter_dict["tfidf_pipeline"]
    HARD_BLOCKLIST = filter_dict["hard_blocklist"]
    RISK_KEYWORDS = filter_dict["risk_keywords"]
    THRESHOLD = filter_dict["threshold"]

    def layer1_check(lyric: str) -> tuple[bool, str]:
        lower = lyric.lower()
        for phrase in HARD_BLOCKLIST:
            if phrase in lower:
                return False, phrase
        return True, ""
        
    def layer3_check(lyric: str) -> tuple[str, float]:
        lower  = lyric.lower().split()
        scores = {}
        for category, keywords in RISK_KEYWORDS.items():
            hits = sum(1 for w in lower if w in keywords)
            scores[category] = hits / max(len(lower), 1)
        best = max(scores, key=scores.get)
        return best, scores[best]
        
    def safety_filter(
        lyric: str,
        instrumentalness: float = 0.0,
        sensitivity_flags: list[str] = [],
    ) -> dict:
        if instrumentalness > 0.6:
            return {"safe": True, "risk_category": "instrumental_skip",
                    "risk_score": 0.0, "block_reason": "", "flagged_terms": []}

        l1_safe, matched = layer1_check(lyric)
        if not l1_safe:
            return {"safe": False, "risk_category": "self_harm",
                    "risk_score": 1.0, "block_reason": "hard_blocklist",
                    "flagged_terms": [matched]}

        risk_score = float(tfidf_pipeline.predict_proba([lyric])[0, 1])
        if risk_score >= THRESHOLD:
            return {"safe": False, "risk_category": "self_harm",
                    "risk_score": risk_score, "block_reason": "tfidf_classifier",
                    "flagged_terms": []}

        sem_cat, sem_score = layer3_check(lyric)
        if sem_cat == "grief_trigger" and "grief_sensitive" in sensitivity_flags:
            return {"safe": False, "risk_category": "grief_trigger",
                    "risk_score": sem_score, "block_reason": "semantic_sensitivity_routing",
                    "flagged_terms": []}

        return {"safe": True, "risk_category": sem_cat if sem_score > 0.1 else "safe",
                "risk_score": risk_score, "block_reason": "", "flagged_terms": []}

    filter_artefact = {"filter_fn": safety_filter}

    therapy_engine = TherapyEngine(backend="mock")
    security = SecurityLayer(
        rate_limit_requests = 30,
        rate_limit_window_s = 60,
        burst_capacity      = 5,
        audit_log_path      = ROOT / "data" / "security" / "audit.log",
        pii_salt            = "harmonicai_demo_salt",
    )
    
    print(_green("Done!\n"))

    run_interactive_request(
        tracks_df       = tracks_df,
        rf_pipeline     = rf_pipeline,
        feature_list    = feature_list,
        filter_artefact = filter_artefact,
        therapy_engine  = therapy_engine,
        security        = security,
        user_mood       = mood_input,
        user_intent     = intent_input,
    )

if __name__ == "__main__":
    main()
