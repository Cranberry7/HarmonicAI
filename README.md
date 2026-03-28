# HarmonicAI — ML-Driven Therapeutic Audio Engine

## Project Structure

```
harmonicai/
│
├── data/
│   ├── raw/            # Original source data (Spotify API pulls, lyric CSVs)
│   ├── synthetic/      # Phase-0 generated datasets for development
│   ├── processed/      # Cleaned, feature-engineered, model-ready datasets
│   └── feedback/       # User session feedback events (Delta-mood labels)
│
├── src/
│   ├── ingestion/      # Phase 1: Data intake, profiling, normalization
│   ├── clustering/     # Phase 2: Acoustic clustering engine (unsupervised)
│   ├── predictor/      # Phase 3: Mood efficacy classifiers (supervised ensemble)
│   ├── frequency/      # Phase 4: Mel-spectrogram generation + CNN analysis
│   ├── nlp/            # Phase 5: Lyrical safety filter (sentiment + NLP)
│   ├── genai/          # Phase 6: Therapy script generation (LLM prompting)
│   ├── feedback/       # Phase 7: Feedback loop + drift detection
│   └── security/       # Phase 8: Threat modelling + compliance utilities
│
├── models/
│   ├── checkpoints/    # Serialized model weights (never commit to git)
│   └── registry/       # Model metadata: version, F1, train date, feature hash
│
├── notebooks/          # Exploratory analysis only — no production logic here
├── configs/            # YAML configs: feature lists, hyperparameter grids
├── tests/              # Unit + integration tests per module
├── logs/               # Training logs, drift alerts
└── scripts/            # CLI runners for pipeline stages
```

## Data Domains

| Domain | Files | Phase |
|---|---|---|
| Acoustic features (metadata) | `data/synthetic/tracks.parquet` | 1–3 |
| User profiles + mood baselines | `data/synthetic/users.parquet` | 1 |
| Session feedback events | `data/feedback/sessions.parquet` | 7 |
| Mel-spectrograms | `data/processed/spectrograms/` | 4 |
| Lyric datasets | `data/raw/lyrics/` | 5 |

## Phase Roadmap

- [x] Phase 0 — Foundations & Dataset Strategy
- [ ] Phase 1 — Data Intake & User Profiling
- [ ] Phase 2 — Acoustic Clustering Engine
- [ ] Phase 3 — Mood Efficacy Predictor
- [ ] Phase 4 — Deep Frequency Analyzer
- [ ] Phase 5 — Lyrical Safety Filter
- [ ] Phase 6 — Dynamic Therapy Generation
- [ ] Phase 7 — Feedback Loop & Continuous Learning
- [ ] Phase 8 — Security & Compliance Engineering
