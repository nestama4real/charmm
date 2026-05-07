# scripts/tokenize_macro.py

"""
Tokenize all Chopin pieces from MAESTRO v3 with the macro tokenizer.
Saves results as .npy arrays of shape (N_measures, 6) per piece.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from charmm.tokenizers.macro import MacroTokenizer


# ── Paths ─────────────────────────────────────────────────────────────────────

MAESTRO_ROOT = Path("data/maestro-v3.0.0-midi/maestro-v3.0.0")
CSV_PATH     = MAESTRO_ROOT / "maestro-v3.0.0.csv"
OUTPUT_ROOT  = Path("data/tokens/macro")


def main() -> None:
    # 1. Load metadata and filter Chopin
    df = pd.read_csv(CSV_PATH)
    df = df[df["canonical_composer"].str.contains("Chopin", case=False, na=False)]
    print(f"Found {len(df)} Chopin pieces in MAESTRO")
    print(df["split"].value_counts().to_string())
    
    # 2. Prepare output directories
    for split in ["train", "validation", "test"]:
        (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)
    
    # 3. Tokenize each piece
    tokenizer = MacroTokenizer()
    failures = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        midi_path = MAESTRO_ROOT / row["midi_filename"]
        out_path  = OUTPUT_ROOT / row["split"] / f"{Path(row['midi_filename']).stem}.npy"
        
        try:
            tokens = tokenizer.tokenize(str(midi_path))
            np.save(out_path, tokens)
        except Exception as e:
            failures.append((row["midi_filename"], str(e)))
    
    # 4. Report
    print(f"\n✓ Tokenized {len(df) - len(failures)} pieces")
    if failures:
        print(f"✗ Failed on {len(failures)}:")
        for fname, err in failures:
            print(f"  {fname}: {err}")


if __name__ == "__main__":
    main()