from charmm.tokenizers.micro import MicroTokenizer

path="data/maestro-v3.0.0-midi/maestro-v3.0.0/2013/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_02_R1_2013_wav--5.midi"

t = MicroTokenizer()
print(f"Vocab size: {t.vocab_size}")

ids = t.tokenize(path)
print(f"Shape: {ids.shape}")
print(f"First 30 tokens: {ids[:30]}")

# Round-trip test
t.detokenize(ids, "/tmp/roundtrip.mid")