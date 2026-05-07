from charmm.tokenizers.macro import MacroTokenizer

path="data/maestro-v3.0.0-midi/maestro-v3.0.0/2013/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_02_R1_2013_wav--5.midi"
t = MacroTokenizer()
tokens = t.tokenize(path)
print(tokens.shape)


from charmm.types import vocab

for row in tokens:
    h_main   = vocab.harm.decode(row[0])
    h_accent = vocab.harm.decode(row[1])
    dens     = vocab.dens.decode(row[2])
    rhy      = vocab.rhy.decode(row[3])
    dyn      = vocab.dyn.decode(row[4])
    pos      = vocab.pos.decode(row[5])
    print(f"{h_main.name:10} {h_accent.name:10} {dens.name:8} {rhy.name:12} {dyn.name:4} {pos.name}")