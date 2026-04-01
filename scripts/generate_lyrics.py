"""
harmonicai/scripts/generate_lyrics.py

Phase 5 — Synthetic Lyric Dataset Generator
=============================================
Generates a synthetic lyric dataset with four risk categories:
  - self_harm      (critical — always block)
  - hopelessness   (high severity — block)
  - grief_trigger  (medium — route by user sensitivity)
  - substance_abuse (medium — block)
  - safe           (no risk)

Why synthetic lyrics?
Real lyric datasets require licensing agreements (Genius API,
AZLyrics). Structurally, synthetic lyrics preserve the statistical
properties that matter for NLP: vocabulary distribution, risk
keyword density, context patterns, and class balance.

In production: replace with Genius API + scraping pipeline.
API endpoint: GET https://api.genius.com/search?q={track_name}

Output:
    data/synthetic/lyrics.csv

Run:
    python scripts/generate_lyrics.py
"""

import numpy as np
import pandas as pd
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
N_TRACKS     = 1500

rng = np.random.default_rng(RANDOM_SEED)

# ─────────────────────────────────────────────
# Lyric Component Libraries
# These are synthetic fragments — not real songs.
# They are constructed to have realistic word distributions
# for each risk category.
# ─────────────────────────────────────────────

SAFE_LINES = [
    "the morning light breaks through the window",
    "we danced until the stars came out",
    "your smile is all I need to carry on",
    "the river runs to the open sea",
    "holding on to every precious moment",
    "the road ahead is long but I am ready",
    "sunshine finds its way through the clouds",
    "you are the reason I keep going",
    "together we can face whatever comes our way",
    "the world is full of wonder if you look",
    "every breath is a chance to start again",
    "I found my peace beside the ocean",
    "the mountains stand tall in the morning mist",
    "love is the anchor through the storm",
    "we rise together in the golden hour",
    "this journey leads somewhere beautiful",
    "the seasons change but you remain",
    "I am learning to be still and breathe",
    "the night gives way to a brand new day",
    "music fills the spaces words cannot reach",
    "your voice echoes in the quiet evening",
    "we built something worth remembering here",
    "the garden grows in its own sweet time",
    "finding my way back to where I belong",
    "grateful for every ordinary moment",
]

HOPELESS_LINES = [
    "there is no way out of this darkness",
    "I have given up trying to feel something",
    "nothing will ever change no matter what I do",
    "the emptiness inside won't go away",
    "I stopped believing things could be different",
    "there is no point in carrying on this way",
    "every door is closed and I am done knocking",
    "the future holds nothing for someone like me",
    "I am invisible to everyone around me",
    "nothing I do will ever be enough",
    "I have lost the will to keep pretending",
    "there is no light at the end of this tunnel",
    "I wake up and I don't know why I bother",
    "everything I touch turns to ash and dust",
    "I am just waiting for it to be over",
    "no one would notice if I disappeared",
    "I am exhausted and I cannot feel anymore",
    "the world doesn't need me and I know it",
    "I don't see the point in any of this",
    "I have given everything I have and still I fail",
]

SELF_HARM_LINES = [
    "I want to disappear from this world for good",
    "the only escape I see is ending this pain",
    "I think about not being here anymore",
    "cutting is the only thing that makes me feel real",
    "I planned a way out and I am not afraid",
    "there is a way to stop this and I know what it is",
    "I have decided I do not want to be alive",
    "the bridge is always there when I walk past it",
    "I cannot live with this anymore and I am serious",
    "I want to hurt myself until I feel nothing",
    "ending it all seems like the only answer left",
    "I wrote a note and left it on the table",
    "I have been thinking about how to do it",
    "the pills are there and the thought won't leave",
    "I said goodbye to everyone without them knowing",
]

GRIEF_LINES = [
    "I still reach for the phone to call you",
    "the chair at the table sits empty now",
    "I keep finding your things around the house",
    "no one told me grief would feel like this",
    "I cannot remember your voice and it terrifies me",
    "the anniversary comes around and I fall apart",
    "I drove past the place where we used to go",
    "learning to live without you is impossible",
    "I bought a card then remembered you were gone",
    "the world moved on but I am still in that day",
    "everyone says it gets easier but they lied",
    "I talk to you even though you cannot hear",
    "the photographs are all I have left of you",
    "grief arrives in waves when I least expect it",
    "I am learning to carry you with me instead",
]

SUBSTANCE_LINES = [
    "I drink until I cannot remember my name",
    "the pills help me forget what I have become",
    "I need another hit to get through the day",
    "I cannot function without the bottle in my hand",
    "the only thing that numbs this is getting high",
    "I woke up again and I don't remember last night",
    "I traded everything I had for one more fix",
    "I hide the bottles so no one knows the truth",
    "the drugs are the only friends I have left",
    "I told myself I would stop and I lied again",
    "I spend my last money on what I shouldn't",
    "the withdrawal is worse than the addiction",
    "I am chasing something I will never catch",
    "it started as relief and became my prison",
    "I blackout and wake up somewhere unfamiliar",
]


def build_lyric(rng, lines: list, n_lines: int = 4) -> str:
    """Randomly sample and join lines to form a lyric snippet."""
    chosen = rng.choice(lines, size=min(n_lines, len(lines)), replace=True)
    return " / ".join(chosen)


def assign_risk_label(
    category: str,
    rng: np.random.Generator
) -> tuple:
    """
    Returns (risk_category, risk_score, should_block).

    risk_score: 0.0 to 1.0 — continuous severity
    should_block: binary decision for the filter to learn

    Noise is added to simulate inter-rater disagreement —
    two clinicians don't always agree on borderline lyrics.
    """

    base_scores = {
        "safe":           0.05,
        "hopelessness":   0.55,
        "grief_trigger":  0.40,
        "substance_abuse":0.50,
        "self_harm":      0.92,
    }

    score = base_scores[category] + rng.normal(0, 0.06)
    score = float(np.clip(score, 0.0, 1.0))

    # Blocking threshold: self_harm always, others above 0.45
    if category == "self_harm":
        block = True
    elif category == "safe":
        block = score > 0.25    # occasionally flag borderline safe content
    else:
        block = score > 0.45

    return score, block


# Category distribution — mirrors realistic lyric catalogue distribution
CATEGORIES    = ["safe", "hopelessness", "grief_trigger", "substance_abuse", "self_harm"]
CAT_WEIGHTS   = [0.60,   0.12,           0.13,            0.09,              0.06]
LINE_POOLS    = {
    "safe":            SAFE_LINES,
    "hopelessness":    HOPELESS_LINES,
    "grief_trigger":   GRIEF_LINES,
    "substance_abuse": SUBSTANCE_LINES,
    "self_harm":       SELF_HARM_LINES,
}


def generate_lyrics(n: int = N_TRACKS) -> pd.DataFrame:
    records = []

    for i in range(n):
        category = str(rng.choice(CATEGORIES, p=CAT_WEIGHTS))
        n_lines  = int(rng.integers(3, 7))
        lyric    = build_lyric(rng, LINE_POOLS[category], n_lines)

        # Occasionally mix categories (realistic — a song can have mixed themes)
        if rng.random() < 0.12 and category != "self_harm":
            secondary = str(rng.choice(
                [c for c in CATEGORIES if c != category and c != "self_harm"],
                p=None
            ))
            extra_line = build_lyric(rng, LINE_POOLS[secondary], n_lines=1)
            lyric = lyric + " / " + extra_line

        risk_score, should_block = assign_risk_label(category, rng)

        records.append({
            "track_id":     f"lyric_{i:04d}_{uuid.uuid4().hex[:6]}",
            "lyric":        lyric,
            "risk_category":category,
            "risk_score":   round(risk_score, 4),
            "should_block": should_block,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 55)
    print("  HarmonicAI — Phase 5: Lyric Dataset Generator")
    print("=" * 55)

    df = generate_lyrics(N_TRACKS)

    # Report
    print(f"\n  Generated {len(df):,} lyric records")
    print(f"\n  Risk category distribution:")
    for cat, count in df["risk_category"].value_counts().items():
        pct = count / len(df)
        bar = "█" * int(pct * 30)
        print(f"    {cat:<18} {bar:<30} {pct:.1%}")

    print(f"\n  Block rate overall: {df['should_block'].mean():.1%}")
    print(f"  Block rate by category:")
    for cat in CATEGORIES:
        sub = df[df["risk_category"] == cat]
        print(f"    {cat:<18} {sub['should_block'].mean():.1%}")

    out_path = OUTPUT_DIR / "lyrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  ✅ Saved to: data/synthetic/lyrics.csv")
