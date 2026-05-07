# scripts/tokenize_micro.py

"""
Tokenize ALL pieces from MAESTRO v3 with the micro tokenizer (REMI+).
Used for pre-training the Chopin micro model on universal piano data.
Saves results as .npy arrays of shape (N_tokens,) per piece.

Resumable: skips pieces already tokenized. Safe to interrupt and resume.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from charmm.tokenizers.micro import MicroTokenizer


# ── Paths ─────────────────────────────────────────────────────────────────────

MAESTRO_ROOT = Path("data/maestro-v3.0.0-midi/maestro-v3.0.0")
CSV_PATH     = MAESTRO_ROOT / "maestro-v3.0.0.csv"
OUTPUT_ROOT  = Path("data/tokens/micro")


def main() -> None:
    # 1. Load metadata
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} pieces in MAESTRO")
    print(df["split"].value_counts().to_string())
    
    # 2. Prepare output directories
    for split in ["train", "validation", "test"]:
        (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)
    
    # 3. Filter out pieces already tokenized
    rows_to_process = []
    skipped = 0
    for _, row in df.iterrows():
        out_path = OUTPUT_ROOT / row["split"] / f"{Path(row['midi_filename']).stem}.npy"
        if out_path.exists():
            skipped += 1
        else:
            rows_to_process.append(row)
    
    print(f"\n→ Already done: {skipped}")
    print(f"→ To process:   {len(rows_to_process)}\n")
    
    if not rows_to_process:
        print("✓ Nothing to do, all pieces already tokenized.")
        return
    
    # 4. Tokenize remaining pieces
    tokenizer = MicroTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}\n")
    failures = []
    
    for row in tqdm(rows_to_process, desc="Tokenizing"):
        midi_path = MAESTRO_ROOT / row["midi_filename"]
        out_path  = OUTPUT_ROOT / row["split"] / f"{Path(row['midi_filename']).stem}.npy"
        
        try:
            tokens = tokenizer.tokenize(midi_path)
            np.save(out_path, tokens)
        except Exception as e:
            failures.append((row["midi_filename"], str(e)))
    
    # 5. Report
    print(f"\n✓ Tokenized {len(rows_to_process) - len(failures)} new pieces")
    if failures:
        print(f"✗ Failed on {len(failures)}:")
        for fname, err in failures[:10]:
            print(f"  {fname}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")


if __name__ == "__main__":
    main()