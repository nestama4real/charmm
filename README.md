# ChARMM — Chopin AutoRegressive Music Model

Generating original piano compositions in the style of Frédéric Chopin, using a hierarchical autoregressive Transformer architecture trained on symbolic MIDI data.

Built as a personal project aside studies at EPFL.

## Overview

ChARMM is a **two-stage hierarchical model** that decouples long-range musical structure from local note-level expressivity. A macro model generates the harmonic and dynamic skeleton of an entire piece; a micro model then realizes the notes of each measure, conditioned on the macro context.

This addresses a fundamental limitation of single-stage music Transformers: short context windows cannot capture full-piece structure (ABA form, recapitulation, dynamic arcs), while expanding the context to thousands of tokens is computationally prohibitive on consumer hardware.

| Stage | Role |
|---|---|
| **Frédéric** (macro) | Generates the harmonic/dynamic structure of the entire piece (~80 measures) |
| **Chopin** (micro) | Realizes the notes of each measure, conditioned on Frédéric's output |

At inference time, Frédéric generates the full macro sequence, then Chopin generates the notes of each measure conditioned on the corresponding macro tokens. The result is detokenized into a `.mid` file.

## Architecture

### Frédéric — macro model

| | |
|---|---|
| **Architecture** | Decoder-only Transformer |
| **Tokenization** | Custom 5-stream macro scheme (see below) |
| **Layers** | 4 |
| **Embedding dim** | 128 |
| **Context length** | 512 tokens (~80 measures = full piece) |
| **Parameters** | ~2M |
| **Training data** | MAESTRO v3 — Chopin only (26.2h) |

### Chopin — micro model

| | |
|---|---|
| **Architecture** | Decoder-only Transformer with macro conditioning |
| **Tokenization** | REMI+ · relative pitch encoding (MidiTok) |
| **Layers** | 6 |
| **Embedding dim** | 256 |
| **Context length** | 256 tokens (1–2 measures) |
| **Parameters** | ~6M |
| **Training data** | MAESTRO v3 (pre-train, 198.7h) → Chopin only (fine-tune, 26.2h) |
| **Conditioning** | Macro tokens prepended as fixed prefix; loss masked on macro portion |

## Tokenization

### Macro tokens (5 streams per measure)

Each measure is encoded as a 5-dimensional vector, one value per stream. The model uses 5 independent embedding tables on input and 5 independent output heads on prediction.

| Stream | Description | Vocab size |
|---|---|---|
| `Harm` | Harmonic color (12 roots × 4 modes) | 48 |
| `Dens` | Note density (sparse / medium / dense) | 3 |
| `RhyContour` | Rhythmic contour (regular / syncopated / fluid) | 3 |
| `Dyn` | Dynamic level (pp → ff) | 6 |
| `Pos` | Relative position in piece, quantised to 21 buckets | 21 |

### Micro tokens

REMI+ scheme via MidiTok with relative pitch encoding (transposition invariance) and quantised velocity (8 levels).

## Project Structure

```
charm/
├── charm/
│   ├── types.py              # Token enums, Vocab, MacroMeasure
│   ├── tokenizers/
│   │   ├── macro.py          # MIDI → macro tokens
│   │   └── micro.py          # MIDI → REMI+ tokens
│   ├── models/
│   │   ├── frederic.py       # Macro Transformer
│   │   └── chopin.py         # Micro Transformer with conditioning
│   ├── loaders/
│   │   ├── dataset.py        # PyTorch Datasets
│   │   └── dataloader.py     # PyTorch DataLoaders
│   └── inference/
│       └── pipeline.py       # Frédéric → Chopin → MIDI assembly
├── scripts/
│   ├── tokenize_macro.py
│   ├── tokenize_micro.py
│   ├── train_frederic.py
│   ├── train_chopin.py
│   └── generate.py
└── data/
    └── maestro-v3.0.0-midi/
```

## Usage

```bash
# Prepare data
python scripts/tokenize_macro.py --maestro_root ./data/maestro-v3.0.0-midi
python scripts/tokenize_micro.py --maestro_root ./data/maestro-v3.0.0-midi

# Train Frédéric on Chopin macro tokens
python scripts/train_frederic.py

# Pre-train Chopin on full MAESTRO, then fine-tune on Chopin
python scripts/train_chopin.py --pretrain
python scripts/train_chopin.py --finetune

# Generate
python scripts/generate.py \
    --frederic checkpoints/frederic.pt \
    --chopin   checkpoints/chopin.pt \
    --temperature 0.9 --top_p 0.9 \
    --output piece.mid
```

## Generation parameters

| Parameter | Default | Description |
|---|---|---|
| `--temperature` | `1.0` | Lower = conservative, higher = creative |
| `--top_p` | `0.9` | Nucleus sampling threshold |
| `--n_measures` | `80` | Length of the generated piece |

## Tech

- **Language:** Python 3.10+
- **Libraries:** PyTorch, MidiTok, pretty_midi, NumPy, FluidSynth
- **Hardware:** RTX 4070 laptop (dev) · RTX 2080 desktop (training, multi-session via checkpointing)

## References

- Hawthorne et al. — [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro), 2019
- Huang et al. — [Music Transformer](https://magenta.tensorflow.org/music-transformer), 2018
- Fradet et al. — [MidiTok](https://miditok.readthedocs.io/), 2021
- Karpathy — [nanoGPT](https://github.com/karpathy/nanoGPT), 2022

## Author

Mattia Nesta

## License

MIT