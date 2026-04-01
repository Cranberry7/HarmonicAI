"""
harmonicai/src/frequency/cnn_analyzer.py

Phase 4 — CNN Frequency Analyzer
==================================
Implements a Convolutional Neural Network for therapeutic audio
classification from Mel-spectrograms.

Since PyTorch is unavailable in this environment, we:
  1. Implement the full CNN architecture in numpy (educational)
  2. Implement a production-equivalent using sklearn's MLP on
     pooled spectrogram features (functional substitute)
  3. Provide the full PyTorch implementation as commented
     production code ready to activate when torch is available

Architecture: TherapyCNN
  Input:   (1, 128, 128)     — (channels, mel_bins, time_frames)
  Conv1:   32 filters, 3×3   — detect local time-frequency patterns
  Pool1:   2×2 MaxPool        — reduce spatial size, build invariance
  Conv2:   64 filters, 3×3   — detect composite patterns
  Pool2:   2×2 MaxPool
  Conv3:   128 filters, 3×3  — detect high-level therapeutic signatures
  Pool3:   2×2 MaxPool
  Flatten: 128 × 16 × 16 = 32,768 → GlobalAvgPool → 128
  FC1:     128 → 64
  FC2:     64  → 3  (harmful, neutral, therapeutic)

Run:
    python src/frequency/cnn_analyzer.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics         import f1_score, classification_report
from sklearn.pipeline        import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
SPEC_DIR     = PROJECT_ROOT / "data" / "processed" / "spectrograms"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"
MODEL_DIR    = PROJECT_ROOT / "models" / "checkpoints"

RANDOM_SEED = 42


# ─────────────────────────────────────────────
# SECTION 1: CNN Architecture (numpy)
# ─────────────────────────────────────────────
# This implements the forward pass manually to show exactly
# what each layer computes. No backpropagation here —
# this is for understanding, not training.

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x). Sets all negative values to zero."""
    return np.maximum(0, x)


def conv2d_single(
    input_map: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 1
) -> np.ndarray:
    """
    2D convolution of a single feature map with a single kernel.

    input_map: (H, W)
    kernel:    (kH, kW)
    returns:   (H_out, W_out)

    The kernel slides across the input, computing a dot product
    at each position. This detects the pattern encoded in the kernel.

    padding='same' (padding=1 with 3×3 kernel) preserves spatial size.
    """

    if padding > 0:
        input_map = np.pad(input_map, padding, mode="constant")

    H, W   = input_map.shape
    kH, kW = kernel.shape
    H_out  = (H - kH) // stride + 1
    W_out  = (W - kW) // stride + 1

    output = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = input_map[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = np.sum(patch * kernel)

    return output


def maxpool2d(feature_map: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    2×2 max pooling. Takes the maximum value in each non-overlapping pool.

    Why max pooling?
    - Reduces spatial size (128×128 → 64×64 after one pool)
    - Builds translation invariance: if a pattern shifts by 1 pixel,
      max pooling still captures it in the same output cell
    - Keeps the strongest activation in each region
    """

    H, W   = feature_map.shape
    H_out  = H // pool_size
    W_out  = W // pool_size
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            patch = feature_map[
                i*pool_size:(i+1)*pool_size,
                j*pool_size:(j+1)*pool_size
            ]
            output[i, j] = patch.max()

    return output


def global_average_pool(feature_maps: np.ndarray) -> np.ndarray:
    """
    Global Average Pooling: averages each feature map to a single value.

    Input:  (C, H, W)  — C channels of H×W feature maps
    Output: (C,)       — one value per channel

    This collapses the spatial dimensions entirely, giving a
    fixed-size vector regardless of input size. It's more
    parameter-efficient than flattening and less prone to overfitting.
    """

    return feature_maps.mean(axis=(1, 2))


def demonstrate_cnn_forward_pass(mel: np.ndarray):
    """
    Demonstrates the CNN forward pass on one spectrogram.
    Uses random initialized kernels — this is not a trained model,
    just a structural demonstration of what each layer does.
    """

    print(f"\n  CNN Forward Pass Demonstration")
    print(f"  Input shape: {mel.shape}  (C, n_mels, T)")

    rng = np.random.default_rng(42)

    # Work with first channel
    x = mel[0]  # (128, 128)
    print(f"\n  Layer 0 — Input feature map: {x.shape}")

    # ── Conv Layer 1: 4 filters 3×3 ─────────────
    # In a trained model these kernels would detect specific patterns:
    #   - A horizontal stripe kernel → detects sustained tones
    #   - A vertical stripe kernel  → detects percussive transients
    #   - A diagonal kernel         → detects frequency glides/slides

    n_filters_1 = 4
    kernels_1   = rng.normal(0, 0.1, (n_filters_1, 3, 3))

    conv1_maps = np.array([
        relu(conv2d_single(x, kernels_1[f])) for f in range(n_filters_1)
    ])  # (4, 128, 128)
    print(f"  Layer 1 — Conv2d (4 filters, 3×3): {conv1_maps.shape}")
    print(f"            Mean activation: {conv1_maps.mean():.4f}  "
          f"(dead neurons: {(conv1_maps == 0).mean():.1%})")

    # ── MaxPool 1: 2×2 ──────────────────────────
    pool1_maps = np.array([maxpool2d(conv1_maps[f]) for f in range(n_filters_1)])
    print(f"  Layer 2 — MaxPool2d (2×2):          {pool1_maps.shape}")

    # ── Conv Layer 2: 8 filters 3×3 ─────────────
    n_filters_2 = 8
    # Each filter in layer 2 is applied to each of the 4 input channels
    conv2_output = np.zeros((n_filters_2, pool1_maps.shape[1], pool1_maps.shape[2]))
    for f2 in range(n_filters_2):
        kernels_2f = rng.normal(0, 0.1, (n_filters_1, 3, 3))
        for f1 in range(n_filters_1):
            conv2_output[f2] += conv2d_single(pool1_maps[f1], kernels_2f[f1])
        conv2_output[f2] = relu(conv2_output[f2])

    print(f"  Layer 3 — Conv2d (8 filters, 3×3): {conv2_output.shape}")

    # ── MaxPool 2: 2×2 ──────────────────────────
    pool2_maps = np.array([maxpool2d(conv2_output[f]) for f in range(n_filters_2)])
    print(f"  Layer 4 — MaxPool2d (2×2):          {pool2_maps.shape}")

    # ── Global Average Pooling ───────────────────
    gap_vector = global_average_pool(pool2_maps)  # (8,)
    print(f"  Layer 5 — GlobalAvgPool:            {gap_vector.shape}")
    print(f"            Values: {gap_vector.round(4)}")

    # ── Fully Connected ──────────────────────────
    W_fc = rng.normal(0, 0.1, (3, 8))  # 3 classes × 8 features
    b_fc = np.zeros(3)
    logits = W_fc @ gap_vector + b_fc

    # Softmax → probabilities
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    pred_class = np.argmax(probs)
    class_names = ["harmful (-1)", "neutral (0)", "therapeutic (+1)"]

    print(f"  Layer 6 — FC(8→3) + Softmax:        {probs.shape}")
    print(f"\n  Prediction probabilities (random weights — untrained):")
    for name, prob in zip(class_names, probs):
        bar = "█" * int(prob * 30)
        print(f"    {name:<20} {bar:<30} {prob:.3f}")
    print(f"  Predicted class: {class_names[pred_class]}")
    print(f"  (Random weights — for architecture illustration only)")


# ─────────────────────────────────────────────
# SECTION 2: Feature Extraction from Spectrograms
# ─────────────────────────────────────────────

def extract_spectrogram_features(mel: np.ndarray) -> np.ndarray:
    """
    Extracts a compact feature vector from a Mel-spectrogram.

    This is the functional substitute for CNN embeddings
    when PyTorch is unavailable. Instead of learned convolutional
    features, we compute hand-crafted statistics that capture
    the same information CNN kernels would learn to detect.

    Features extracted (total: 52 dimensions):
        Frequency band statistics (4 bands × 4 stats = 16)
        Temporal statistics per band (4 bands × 3 = 12)
        Spectral features (8)
        Rhythm features (4)
        Global statistics (12)
    """

    x = mel[0]  # (128, T) — remove channel dim
    n_mels, T = x.shape

    feats = []

    # ── Frequency band statistics ────────────────
    # Split into 4 therapeutic bands:
    #   Sub-bass (0-20): binaural beats, earth tones
    #   Bass (20-40):    warmth, grounding
    #   Mid (40-80):     melody, voice
    #   High (80-128):   brightness, air
    bands = [(0, 16), (16, 40), (40, 80), (80, 128)]
    for b_low, b_high in bands:
        band = x[b_low:b_high, :]
        feats.extend([
            band.mean(),                   # mean energy in band
            band.std(),                    # energy variance in band
            band.max(),                    # peak energy
            (band > 0.5).mean(),           # fraction of time at high energy
        ])

    # ── Temporal statistics per band ─────────────
    for b_low, b_high in bands:
        band = x[b_low:b_high, :].mean(axis=0)   # average across freq → (T,)
        feats.extend([
            np.diff(band).std(),                  # temporal variability (0=constant drone)
            np.percentile(band, 90),              # high-energy moments
            np.percentile(band, 10),              # low-energy moments (silences)
        ])

    # ── Spectral features ─────────────────────────
    energy_per_bin = x.mean(axis=1) + 1e-8
    total_energy   = energy_per_bin.sum()

    # Spectral centroid (energy-weighted mean frequency)
    mel_bins = np.arange(n_mels)
    centroid = np.average(mel_bins, weights=energy_per_bin) / n_mels

    # Spectral spread (energy-weighted std)
    spread = np.sqrt(np.average((mel_bins - centroid * n_mels) ** 2,
                                weights=energy_per_bin)) / n_mels

    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumulative = np.cumsum(energy_per_bin) / total_energy
    rolloff = np.searchsorted(cumulative, 0.85) / n_mels

    # Spectral flatness (how noise-like vs tonal)
    flatness = (np.exp(np.log(energy_per_bin + 1e-8).mean()) /
                (energy_per_bin.mean() + 1e-8))

    feats.extend([centroid, spread, rolloff, flatness,
                  total_energy, x.std(), x.max(), x.min()])

    # ── Rhythm features ───────────────────────────
    # Energy flux: how much the total energy changes frame to frame
    frame_energy = x.sum(axis=0)                        # (T,)
    energy_diff  = np.diff(frame_energy)
    onset_strength   = np.maximum(0, energy_diff)       # positive = onset

    feats.extend([
        onset_strength.mean(),               # average onset strength
        (onset_strength > onset_strength.mean() + onset_strength.std()).mean(),  # onset rate
        frame_energy.std() / (frame_energy.mean() + 1e-8),  # energy variability
        np.percentile(frame_energy, 95),     # peak loudness moment
    ])

    # ── Global statistics ─────────────────────────
    feats.extend([
        x.mean(), x.std(), x.max(), x.min(),
        np.percentile(x, 25), np.percentile(x, 75),
        (x > 0.8).mean(),    # fraction of high-energy cells
        (x < 0.1).mean(),    # fraction of silence/near-silence
        x[:32, :].mean(),    # low-freq half mean
        x[64:, :].mean(),    # high-freq half mean
        x[:, :T//2].mean(),  # first half energy
        x[:, T//2:].mean(),  # second half energy (track evolution)
    ])

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────
# SECTION 3: Load Dataset & Train
# ─────────────────────────────────────────────

def load_spectrogram_dataset(labels_df: pd.DataFrame) -> tuple:
    """
    Loads all spectrograms and extracts feature vectors.
    Returns (X, y) ready for sklearn classifiers.
    """

    X_list, y_list = [], []
    missing = 0

    print(f"  Extracting features from {len(labels_df)} spectrograms...")

    for _, row in labels_df.iterrows():
        spec_path = PROJECT_ROOT / row["spec_path"]
        if not spec_path.exists():
            missing += 1
            continue

        mel   = np.load(spec_path)
        feats = extract_spectrogram_features(mel)
        X_list.append(feats)
        y_list.append(int(row["efficacy_label"]))

    if missing > 0:
        print(f"  ⚠  {missing} spectrogram files not found — skipped")

    X = np.vstack(X_list)
    y = np.array(y_list)

    print(f"  Feature matrix: {X.shape}  ({X.shape[1]} hand-crafted features)")
    print(f"  Labels:         {y.shape}")

    return X, y


def train_cnn_substitute(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Trains an MLP classifier on spectrogram features.

    This is the functional substitute for a trained CNN:
      - CNN: learns convolutional filters → pools → FC layers
      - MLP: receives hand-crafted features → FC layers

    The MLP architecture mirrors the CNN's FC head:
        52 features → 128 → 64 → 3 classes

    hidden_layer_sizes=(128, 64) matches our CNN's FC layers.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,          # L2 regularization
            batch_size=32,
            learning_rate="adaptive",
            max_iter=500,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    print(f"\n  Training MLP on spectrogram features (CNN substitute)...")
    print(f"  Architecture: {X.shape[1]} → 128 → 64 → 3")

    fold_f1s = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        fold_f1s.append(f1)

    mean_f1 = np.mean(fold_f1s)
    std_f1  = np.std(fold_f1s)

    print(f"\n  5-fold CV Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # Train on full data for production
    pipeline.fit(X, y)
    y_pred_train = pipeline.predict(X)

    print(f"\n  Training set classification report:")
    print(classification_report(
        y, y_pred_train,
        target_names=["harmful (-1)", "neutral (0)", "therapeutic (+1)"],
        zero_division=0
    ))

    return pipeline, mean_f1


# ─────────────────────────────────────────────
# SECTION 4: What the Full PyTorch CNN Looks Like
# ─────────────────────────────────────────────

PYTORCH_CNN_CODE = '''
# ─────────────────────────────────────────────────────────
# PRODUCTION CNN — activate when torch + librosa available
# Place in: src/frequency/therapy_cnn_torch.py
# ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F

class TherapyCNN(nn.Module):
    """
    3-layer CNN for therapeutic audio classification.
    Input:  (batch, 1, 128, 128)  — Mel-spectrograms
    Output: (batch, 3)            — class logits
    """

    def __init__(self, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # (32, 64, 64)
            nn.Dropout2d(dropout / 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # (64, 32, 32)
            nn.Dropout2d(dropout / 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # (128, 16, 16)
        )

        # Global Average Pooling: (128, 16, 16) → (128,)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)    # flatten (batch, 128)
        return self.classifier(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Training setup
# model     = TherapyCNN(n_classes=3).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
'''


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 58)
    print("  HarmonicAI — Phase 4: CNN Frequency Analyzer")
    print("=" * 58)

    # ── Load labels ────────────────────────────────────────
    labels_path = OUTPUT_DIR / "spectrogram_labels.csv"
    if not labels_path.exists():
        print("\n  ⚠  spectrogram_labels.csv not found.")
        print("  Run first: python src/frequency/spectrogram_dataset.py")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path)
    print(f"\n  Loaded {len(labels_df)} spectrogram labels from:")
    print(f"  → data/processed/spectrogram_labels.csv")

    # ── CNN forward pass demo ──────────────────────────────
    print(f"\n[1/4] Demonstrating CNN forward pass architecture...")
    sample_mel = np.load(PROJECT_ROOT / labels_df.iloc[0]["spec_path"])
    demonstrate_cnn_forward_pass(sample_mel)

    # ── Extract features ───────────────────────────────────
    print(f"\n[2/4] Extracting spectrogram features...")
    X, y = load_spectrogram_dataset(labels_df)

    # ── Train MLP (CNN substitute) ─────────────────────────
    print(f"\n[3/4] Training MLP classifier on spectrogram features...")
    cnn_model, cv_f1 = train_cnn_substitute(X, y)

    # ── Print PyTorch architecture ─────────────────────────
    print(f"\n[4/4] Production PyTorch CNN architecture:")
    print(f"  (Saved to: models/checkpoints/therapy_cnn_architecture.py)")
    arch_path = MODEL_DIR / "therapy_cnn_architecture.py"
    with open(arch_path, "w") as f:
        f.write(PYTORCH_CNN_CODE)
    print(f"  → Activate when torch is available in your environment")
    print(f"  → pip install torch torchaudio librosa")

    # ── Save model ─────────────────────────────────────────
    cnn_path = MODEL_DIR / "cnn_substitute.pkl"
    with open(cnn_path, "wb") as f:
        pickle.dump(cnn_model, f)

    # ── Save feature metadata ──────────────────────────────
    meta = {
        "model_type":    "MLP on hand-crafted spectrogram features",
        "n_features":    int(X.shape[1]),
        "cv_f1_macro":   round(float(cv_f1), 4),
        "architecture":  "52 → 128 → 64 → 3",
        "production_replacement": "TherapyCNN (PyTorch) — see therapy_cnn_architecture.py",
        "input_shape":   "(1, 128, 128)",
        "feature_groups": [
            "frequency_band_stats (16)",
            "temporal_stats_per_band (12)",
            "spectral_features (8)",
            "rhythm_features (4)",
            "global_statistics (12)"
        ]
    }
    import json
    with open(OUTPUT_DIR / "cnn_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n\n✅ Phase 4 complete.")
    print(f"   Outputs:")
    print(f"   data/processed/spectrograms/          ← 500 .npy spectrogram files")
    print(f"   data/processed/spectrogram_labels.csv ← labels + metadata")
    print(f"   data/processed/cnn_model_meta.json    ← model metadata")
    print(f"   models/checkpoints/cnn_substitute.pkl ← trained MLP (CNN substitute)")
    print(f"   models/checkpoints/therapy_cnn_architecture.py ← PyTorch CNN (production)")
    print(f"\nNext step → Phase 5: python src/nlp/safety_filter.py\n")
