# ChARMM — Chopin AutoRegressive Music Model
Generating original piano compositions in the style of Frédéric Chopin, using a GPT-style Transformer trained on symbolic MIDI data.
Built as a personal project aside studies at EPFL.

## Overview
ChARMM frames music generation as a **next-token prediction** problem. MIDI files from the [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) dataset are tokenized into discrete event sequences using the REMI+ scheme, and a decoder-only Transformer is trained to model their distribution — learning harmony, voice leading, and phrasing from data alone, with no hand-crafted music theory rules.

At inference time, tokens are sampled autoregressively and converted back to a `.mid` file.

## Model

| | |
|---|---|
| **Architecture** | Decoder-only Transformer (GPT-style) |
| **Tokenization** | REMI+ · relative pitch encoding (MidiTok) |
| **Layers** | 6 |
| **Attention heads** | 8 |
| **Embedding dim** | 512 |
| **Context length** | 1024 tokens |
| **Parameters** | ~20M |
| **Training data** | MAESTRO v3 — Chopin only (~3h of solo piano) |

## Project Structure
```
charmm/
├── train.py                  # Training loop
├── generate.py               # Autoregressive generation + MIDI export
├── data/
│   ├── filter_maestro.py     # Filter MAESTRO to Chopin pieces
│   └── tokenize.py           # MIDI → REMI+ token sequences
└── model/
    ├── transformer.py        # Decoder-only Transformer
    └── attention.py          # Multi-head self-attention
```

## Usage

```bash
# Prepare data
python data/filter_maestro.py --maestro_root ./maestro-v3.0.0
python data/tokenize.py --midi_dir ./data/chopin

# Train
python train.py --context_len 1024 --batch_size 16 --lr 3e-4

# Generate
python generate.py --checkpoint checkpoints/charmm.pt --temperature 0.9 --top_p 0.9 --output piece.mid
```

## Generation parameters

| Parameter | Default | Description |
|---|---|---|
| `--temperature` | `1.0` | Lower = conservative, higher = creative |
| `--top_p` | `0.9` | Nucleus sampling — filters out unlikely tokens |
| `--length` | `512` | Number of tokens to generate |
| `--prompt` | `BAR POS_0` | Token sequence used as generation seed |

## Tech
- **Language:** Python
- **Libraries:** PyTorch, MidiTok, pandas, pretty_midi
- **Hardware:** RTX 4070 (dev) · EPFL cluster A100 80G (training)

## References
- Hawthorne et al. — [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro), 2019
- Huang et al. — [Music Transformer](https://magenta.tensorflow.org/music-transformer), 2018
- Fradet et al. — [MidiTok](https://miditok.readthedocs.io/), 2021
- Karpathy — [nanoGPT](https://github.com/karpathy/nanoGPT), 2022

## Author
Mattia Nesta

## License
MIT
