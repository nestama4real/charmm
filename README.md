# CLAIR
### Chopin-Like Autoregressive Inference and Representation

CLAIR is a Music Language Model that generates original piano compositions in the style of Frédéric Chopin. It frames music generation as a next-token prediction problem: MIDI files are tokenized into discrete event sequences, and a GPT-style Transformer is trained to model their distribution — learning harmony, rhythm, and phrasing from data alone, without hand-crafted music theory rules.

Trained on the Chopin subset of the [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) dataset, CLAIR generates `.mid` files that can be rendered to audio or opened in any DAW.

---

## Example output

```bash
python generate.py \
  --checkpoint checkpoints/clair.pt \
  --prompt "BAR POS_0 PITCH_60" \
  --temperature 0.9 \
  --top_p 0.9 \
  --length 512 \
  --output nocturne.mid
```

---

## How it works

Each MIDI file is converted into a flat sequence of discrete tokens using the REMI+ scheme ([MidiTok](https://miditok.readthedocs.io/)):

```
BAR  POS_0  PITCH_64  VELOCITY_80  DURATION_quarter  POS_4  PITCH_67  ...
```

A decoder-only Transformer is trained to predict the next token at every position. At inference time, tokens are sampled autoregressively — one at a time — and appended to the growing sequence, producing a composition of arbitrary length.

Relative pitch encoding is used instead of absolute pitch values, making the model invariant to transposition and improving generalization across keys.

---

## Model

| | |
|---|---|
| Architecture | Decoder-only Transformer |
| Tokenization | REMI+ · relative pitch encoding |
| Layers | 6 |
| Attention heads | 8 |
| Embedding dim | 512 |
| Context length | 1024 tokens |
| Parameters | ~20M |
| Training data | MAESTRO v3 — Chopin only |
| Framework | PyTorch |

---

## Installation

```bash
git clone https://github.com/nestama4real/CLAIR
cd clair
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, MidiTok, pandas, pretty_midi.

---

## Usage

### 1. Prepare the data

Download [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro), then run:

```bash
python data/filter_maestro.py --maestro_root ./maestro-v3.0.0
python data/tokenize.py --midi_dir ./data/chopin
```

### 2. Train

```bash
python train.py --context_len 1024 --batch_size 16 --lr 3e-4
```

For a full run on an A100:

```bash
python train.py --context_len 4096 --batch_size 64
```

### 3. Generate

```bash
python generate.py \
  --checkpoint checkpoints/clair.pt \
  --temperature 1.0 \
  --top_p 0.9 \
  --length 512 \
  --output output.mid
```

| Parameter | Default | Description |
|---|---|---|
| `--temperature` | `1.0` | Sampling temperature. Lower = conservative, higher = creative. |
| `--top_p` | `0.9` | Nucleus sampling cutoff. Filters out low-probability tokens. |
| `--length` | `512` | Number of tokens to generate. |
| `--prompt` | `BAR POS_0` | Token sequence used as generation seed. |

---

## Project structure

```
clair/
├── data/
│   ├── filter_maestro.py   # Filter MAESTRO to Chopin pieces
│   └── tokenize.py         # MIDI → REMI+ token sequences
├── model/
│   ├── transformer.py      # Decoder-only Transformer
│   └── attention.py        # Multi-head self-attention
├── train.py                # Training loop
├── generate.py             # Autoregressive generation
├── requirements.txt
└── README.md
```

---

## References

- Hawthorne et al. — [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro), 2019
- Huang et al. — [Music Transformer](https://magenta.tensorflow.org/music-transformer), 2018
- Fradet et al. — [MidiTok](https://miditok.readthedocs.io/), 2021
- Karpathy — [nanoGPT](https://github.com/karpathy/nanoGPT), 2022

---

## License

MIT © Mattia Nesta
