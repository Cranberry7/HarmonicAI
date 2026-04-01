"""
harmonicai/src/frequency/spectrogram_dataset.py

Phase 4 — Synthetic Spectrogram Dataset Generator
===================================================
Generates synthetic Mel-spectrograms that have the structural
properties of real therapeutic audio — without requiring
actual MP3/WAV files.

Why synthetic spectrograms?
Real audio is unavailable in this environment. But more importantly,
this teaches you exactly what signal patterns a CNN must learn to
recognize — because we're building them deliberately.

Each spectrogram is built by composing known acoustic patterns:
  - Binaural beats       → horizontal stripe at specific Mel bin
  - Rhythmic percussion  → periodic vertical spikes (tempo-driven)
  - Harmonic content     → parallel horizontal lines (overtone series)
  - Ambient noise floor  → low-level Gaussian background
  - Spectral centroid    → energy concentration high vs low frequency

The label assigned to each spectrogram uses the same oracle
logic from Phase 0, now applied to the frequency domain.

Run:
    python src/frequency/spectrogram_dataset.py

Output:
    data/processed/spectrograms/   ← .npy files, one per track sample
    data/processed/spectrogram_labels.csv  ← label + metadata per file
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.frequency.mel_transform import audio_to_melspectrogram, SR, N_MELS

PROJECT_ROOT  = Path(__file__).parent.parent.parent
SPEC_DIR      = PROJECT_ROOT / "data" / "processed" / "spectrograms"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "processed"
SPEC_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
N_SAMPLES     = 500       # spectrograms to generate (subset of 5000 tracks)
DURATION_SEC  = 30.0      # analyze first 30 seconds (standard in audio ML)
TARGET_FRAMES = 128       # fixed time dimension for CNN (pad/crop to this)


# ─────────────────────────────────────────────
# Therapeutic Pattern Synthesizers
# ─────────────────────────────────────────────
# Each function synthesizes a type of acoustic pattern
# directly in the time domain, which is then transformed
# to a Mel-spectrogram by the standard pipeline.

def synthesize_binaural_beat(
    rng: np.random.Generator,
    beat_freq: float = 40.0,
    carrier_freq: float = 200.0,
    duration: float = DURATION_SEC,
) -> np.ndarray:
    """
    Synthesizes a binaural beat signal.

    A binaural beat is produced when two pure tones with slightly
    different frequencies are presented to each ear:
      Left ear:  carrier_freq Hz          (e.g. 200 Hz)
      Right ear: carrier_freq + beat_freq (e.g. 240 Hz)

    The brain perceives a third 'phantom' beat at the difference:
      40 Hz → gamma waves → associated with focus and concentration
      10 Hz → alpha waves → associated with relaxed alertness
       4 Hz → theta waves → associated with meditation and sleep

    For mono analysis we simulate the beat as amplitude modulation:
      signal = carrier × (1 + depth × cos(2π × beat_freq × t))
    """

    t = np.linspace(0, duration, int(SR * duration))
    depth = 0.4 + rng.uniform(-0.1, 0.1)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    modulation = 1.0 + depth * np.cos(2 * np.pi * beat_freq * t)
    signal = (carrier * modulation).astype(np.float32)
    noise = (rng.normal(0, 0.02, len(t))).astype(np.float32)
    return signal + noise


def synthesize_rhythmic_track(
    rng: np.random.Generator,
    tempo_bpm: float = 72.0,
    energy: float = 0.4,
    duration: float = DURATION_SEC,
) -> np.ndarray:
    """
    Synthesizes a rhythmic track with harmonic content.

    Consists of:
      - Fundamental frequency (melody/chord)
      - Harmonic overtones (natural instrument timbre)
      - Percussive impulses at beat intervals (rhythm)
      - Background noise floor
    """

    t = np.linspace(0, duration, int(SR * duration))
    n_samples = len(t)

    # Fundamental tone (randomized pitch in therapeutic range)
    fund_freq = rng.uniform(120, 400)
    signal = 0.4 * np.sin(2 * np.pi * fund_freq * t)

    # Add overtones (harmonic series)
    for harmonic in [2, 3, 4]:
        amp = 0.2 / harmonic
        signal += amp * np.sin(2 * np.pi * fund_freq * harmonic * t)

    # Percussive beats at tempo intervals
    beat_interval = int(SR * 60.0 / tempo_bpm)
    beat_positions = np.arange(0, n_samples, beat_interval)

    for pos in beat_positions:
        end = min(pos + int(SR * 0.05), n_samples)   # 50ms impulse
        decay = np.exp(-np.linspace(0, 8, end - pos))
        signal[pos:end] += energy * decay

    noise = rng.normal(0, 0.01, n_samples)
    signal = (signal + noise).astype(np.float32)

    # Normalize
    max_amp = np.abs(signal).max()
    if max_amp > 0:
        signal = signal / max_amp * 0.8

    return signal


def synthesize_ambient_track(
    rng: np.random.Generator,
    energy: float = 0.2,
    duration: float = DURATION_SEC,
) -> np.ndarray:
    """
    Synthesizes an ambient/drone track.
    Low energy, slowly evolving, no percussive content.
    Associated with sleep_induction and anxiety_relief.
    """

    t = np.linspace(0, duration, int(SR * duration))

    # Slow-moving fundamental
    fund_freq = rng.uniform(55, 110)

    # Slow LFO modulation (0.1 Hz = 10 second cycle)
    lfo = 0.5 + 0.3 * np.sin(2 * np.pi * 0.1 * t)

    signal = lfo * np.sin(2 * np.pi * fund_freq * t)
    signal += 0.2 * lfo * np.sin(2 * np.pi * fund_freq * 2 * t)

    # Pink noise background (1/f — more natural than white noise)
    white = rng.normal(0, 1, len(t))
    # Simple 1/f approximation via cumulative sum + normalization
    pink = np.cumsum(white)
    pink = pink / (np.abs(pink).max() + 1e-8)
    signal += energy * 0.1 * pink

    signal = signal.astype(np.float32)
    max_amp = np.abs(signal).max()
    if max_amp > 0:
        signal = signal / max_amp * 0.5

    return signal


def synthesize_high_energy_track(
    rng: np.random.Generator,
    tempo_bpm: float = 128.0,
    duration: float = DURATION_SEC,
) -> np.ndarray:
    """
    Synthesizes a high-energy, fast-tempo track.
    Representative of tracks that are harmful for anxiety states
    but potentially useful for mood_uplift.
    """

    t = np.linspace(0, duration, int(SR * duration))
    n_samples = len(t)

    # Multiple mid-high frequency tones
    signal = np.zeros(n_samples)
    for freq in [320, 480, 640]:
        signal += 0.3 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))

    # Fast, hard beats
    beat_interval = int(SR * 60.0 / tempo_bpm)
    for pos in np.arange(0, n_samples, beat_interval):
        end = min(pos + int(SR * 0.02), n_samples)
        signal[pos:end] += 0.8 * np.exp(-np.linspace(0, 15, end - pos))

    # Broadband noise (characteristic of distorted/compressed audio)
    signal += rng.normal(0, 0.05, n_samples)
    signal = (signal / (np.abs(signal).max() + 1e-8) * 0.9).astype(np.float32)

    return signal


# ─────────────────────────────────────────────
# Oracle Label for Spectrograms
# ─────────────────────────────────────────────

def assign_spectrogram_label(
    track_type: str,
    intent: str,
    mel: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.10,
) -> int:
    """
    Assigns a therapeutic efficacy label based on the track type
    and the requesting user's intent.

    Uses the same oracle logic as Phase 0, but now adds a
    frequency-domain feature: spectral_center — the energy-weighted
    mean Mel bin index. Tracks with low spectral centers have
    most energy in low-frequency bins (calming). High centers
    indicate high-frequency energy dominance (stimulating).
    """

    # Compute spectral center from spectrogram
    mel_bins    = np.arange(N_MELS)
    energy_sum  = mel.sum(axis=1) + 1e-8    # sum across time per frequency bin
    spectral_center = float(np.average(mel_bins, weights=energy_sum)) / N_MELS

    score = 0.0

    type_intent_map = {
        ("binaural",  "deep_focus"):       +2.0,
        ("binaural",  "anxiety_relief"):   +1.5,
        ("binaural",  "sleep_induction"):  +1.0,
        ("ambient",   "sleep_induction"):  +2.0,
        ("ambient",   "anxiety_relief"):   +1.5,
        ("ambient",   "grief_processing"): +1.0,
        ("rhythmic",  "mood_uplift"):      +1.5,
        ("rhythmic",  "deep_focus"):       +0.8,
        ("rhythmic",  "grief_processing"): +0.5,
        ("high_energy", "mood_uplift"):    +0.5,
        ("high_energy", "anxiety_relief"): -2.0,
        ("high_energy", "sleep_induction"):-2.0,
    }

    score += type_intent_map.get((track_type, intent), 0.0)

    # Spectral center penalty for anxiety/sleep intents
    if intent in ("anxiety_relief", "sleep_induction") and spectral_center > 0.5:
        score -= 1.0

    # Add self-report noise
    score += rng.normal(0, noise_std)

    if score >= 1.2:
        return 1
    elif score <= -0.2:
        return -1
    else:
        return 0


# ─────────────────────────────────────────────
# Fixed-Width Padding / Cropping
# ─────────────────────────────────────────────

def normalize_frames(mel: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Pads or crops spectrogram to a fixed time dimension.

    Why fixed width?
    CNNs require fixed-size inputs. Tracks have different lengths.
    Standard approach: analyze the first 30 seconds, then:
      - If T < target_frames: zero-pad on the right
      - If T > target_frames: crop from the center (avoids intro/outro bias)
    """

    T = mel.shape[1]
    if T >= target_frames:
        start = (T - target_frames) // 2
        return mel[:, start:start + target_frames]
    else:
        pad_width = target_frames - T
        return np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")


# ─────────────────────────────────────────────
# Dataset Generator
# ─────────────────────────────────────────────

INTENTS = [
    "anxiety_relief", "sleep_induction", "deep_focus",
    "mood_uplift", "grief_processing"
]
INTENT_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

TRACK_TYPES   = ["binaural", "rhythmic", "ambient", "high_energy"]
TRACK_WEIGHTS = [0.20,       0.45,       0.25,      0.10]


def generate_spectrogram_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    """
    Generates n synthetic spectrograms and saves them as .npy files.

    Each file:
        Path:  data/processed/spectrograms/{track_id}.npy
        Shape: (1, N_MELS, TARGET_FRAMES) = (1, 128, 128)
        The leading 1 is the channel dimension — CNNs expect (C, H, W)

    Returns a DataFrame of metadata for all generated spectrograms.
    """

    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    print(f"  Generating {n} synthetic spectrograms...")
    print(f"  Output directory: {SPEC_DIR}")
    print(f"  Shape per file: (1, {N_MELS}, {TARGET_FRAMES})")

    for i in range(n):
        track_id   = f"synth_spec_{i:04d}"
        track_type = rng.choice(TRACK_TYPES, p=TRACK_WEIGHTS)
        intent     = rng.choice(INTENTS, p=INTENT_WEIGHTS)

        # Synthesize audio signal based on track type
        if track_type == "binaural":
            beat_freq = rng.choice([4.0, 10.0, 40.0])    # theta, alpha, gamma
            signal = synthesize_binaural_beat(rng, beat_freq=beat_freq)
        elif track_type == "rhythmic":
            tempo = rng.uniform(55, 130)
            energy = rng.uniform(0.2, 0.6)
            signal = synthesize_rhythmic_track(rng, tempo_bpm=tempo, energy=energy)
        elif track_type == "ambient":
            signal = synthesize_ambient_track(rng, energy=rng.uniform(0.1, 0.3))
        else:  # high_energy
            signal = synthesize_high_energy_track(rng, tempo_bpm=rng.uniform(120, 160))

        # Transform to Mel-spectrogram
        mel = audio_to_melspectrogram(signal)

        # Normalize to fixed frame count
        mel = normalize_frames(mel, TARGET_FRAMES)

        # Add channel dimension: (128, 128) → (1, 128, 128)
        mel = mel[np.newaxis, :, :]

        # Compute spectral features for labeling
        label = assign_spectrogram_label(
            track_type, intent, mel[0], rng
        )

        # Save spectrogram
        spec_path = SPEC_DIR / f"{track_id}.npy"
        np.save(spec_path, mel)

        records.append({
            "track_id":         track_id,
            "spec_path":        str(spec_path.relative_to(PROJECT_ROOT)),
            "track_type":       track_type,
            "session_intent":   intent,
            "efficacy_label":   label,
            "shape":            str(mel.shape),
        })

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n} spectrograms generated...")

    df = pd.DataFrame(records)

    # Report label distribution
    print(f"\n  Label distribution:")
    dist = df["efficacy_label"].value_counts(normalize=True).sort_index()
    for label, pct in dist.items():
        bar = "█" * int(pct * 30)
        print(f"    {label:+d}: {bar:<30} {pct:.1%}")

    print(f"\n  Track type distribution:")
    for tt, pct in df["track_type"].value_counts(normalize=True).items():
        print(f"    {tt:<15} {pct:.1%}")

    return df


if __name__ == "__main__":
    print("=" * 58)
    print("  HarmonicAI — Phase 4: Spectrogram Dataset Generator")
    print("=" * 58)

    df = generate_spectrogram_dataset(N_SAMPLES)

    # Save metadata
    out_path = OUTPUT_DIR / "spectrogram_labels.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  ✅ Dataset generated")
    print(f"     {N_SAMPLES} .npy files → {SPEC_DIR}")
    print(f"     Metadata   → {out_path}")

    # Validate one saved file
    sample = np.load(SPEC_DIR / "synth_spec_0000.npy")
    print(f"\n  Validation — loaded synth_spec_0000.npy:")
    print(f"     Shape:  {sample.shape}  (C, n_mels, T)")
    print(f"     Range:  [{sample.min():.4f}, {sample.max():.4f}]")
    print(f"     dtype:  {sample.dtype}")
