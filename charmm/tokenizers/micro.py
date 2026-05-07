import miditok as mt
import numpy as np

class MicroTokenizer:
    """
    Wrapper around MidiTok's REMI+ tokenizer for ChARMM's micro model.
    
    Encodes a MIDI file as a flat sequence of integer token IDs capturing
    note-level detail (pitch, velocity, duration, position, pedal).
    """
    def __init__(self):
        config = mt.TokenizerConfig(
            pitch_range=(21, 108),
            beat_res={(0, 4): 16, (4, 12): 8},
            num_velocities=16,
            use_chords=False,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=False,
            use_pedal=True,
            use_pitch_bends=False,
        )
        self._tokenizer = mt.REMI(config)
    
    def tokenize(self, path: str) -> np.ndarray:
        """MIDI file → (N,) int32 array of token IDs."""
        tokens = self._tokenizer(str(path))
        # MidiTok returns either a TokSequence or a list of TokSequence.
        # For solo piano (single track), unwrap the list.
        if isinstance(tokens, list):
            tokens = tokens[0]
        return np.array(tokens.ids, dtype=np.int32)

    def detokenize(self, ids: np.ndarray, output_path: str) -> None:
        """Token IDs → .mid file."""
        score = self._tokenizer.decode([ids.tolist()])
        score.dump_midi(str(output_path))

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer.vocab)