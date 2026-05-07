import pretty_midi as pm
import numpy as np
from charmm.types import *
from charmm.utils.harmonic_extractor import HarmonicExtractor

class MacroTokenizer:
    def __init__(self):
        self._harm_extractor = HarmonicExtractor()

    def _load(self, path: str) -> pm.PrettyMIDI:
        return pm.PrettyMIDI(path)

    def _get_measures(self, midi: pm.PrettyMIDI) -> np.ndarray:
        """
        Returns a list of (start, end) timestamps in seconds for each measure.
        Uses pretty_midi's get_downbeats() which properly handles tempo and
        time signature changes.
        """
        downbeats = midi.get_downbeats()
        
        if len(downbeats) == 0:
            return np.empty((0, 2), dtype=np.float64)

        
        end_time = midi.get_end_time()
        starts = downbeats
        ends = np.append(downbeats[1:], end_time)
        
        return np.stack([starts, ends], axis=1)

    def _extract_harm(self, midi: pm.PrettyMIDI, start: float, end: float):
        return self._harm_extractor.extract(midi, start, end)

    def _extract_dens(self, midi: pm.PrettyMIDI, start: float, end: float) -> Token:
        """Note density per measure → DensToken bucket."""
        n_notes = sum(
            1
            for inst in midi.instruments
            for note in inst.notes
            if start <= note.start < end
        )
        
        duration = end - start
        if duration <= 0:
            return DensToken.SPARSE
        
        notes_per_second = n_notes / duration
        
        # Thresholds calibrated for solo piano:
        # sparse: < 4 notes/s (slow melodies, sustained chords)
        # medium: 4-10 notes/s (typical accompaniment)
        # dense:  > 10 notes/s (fast passages, runs)
        if notes_per_second < 4:   return DensToken.SPARSE
        if notes_per_second < 10:  return DensToken.MEDIUM
        return DensToken.DENSE

    def _extract_rhycontour(self, midi: pm.PrettyMIDI, start: float, end: float) -> Token:
        """Rhythmic contour: regular / syncopated / fluid."""
        notes = sorted(
            [n for inst in midi.instruments for n in inst.notes
            if start <= n.start < end],
            key=lambda n: n.start
        )
        
        if len(notes) < 3:
            return RhyContourToken.REGULAR  # not enough info → default
        
        # 1. IOI uniformity
        iois = np.diff([n.start for n in notes])
        ioi_cv = np.std(iois) / (np.mean(iois) + 1e-9)  # coefficient of variation
        mean_ioi = np.mean(iois)
        
        # 2. Beat alignment
        beats = midi.get_beats()
        beats_in_measure = beats[(beats >= start) & (beats < end)]
        if len(beats_in_measure) == 0:
            return RhyContourToken.REGULAR
        
        beat_duration = (end - start) / max(len(beats_in_measure), 1)
        tolerance = beat_duration * 0.1  # 10% tolerance for "on the beat"
        
        on_beat = sum(
            1 for n in notes
            if any(abs(n.start - b) < tolerance for b in beats_in_measure)
        )
        on_beat_ratio = on_beat / len(notes)
        
        # 3. Classification
        # Fluid: dense + uniform IOI (continuous flow of fast notes)
        if mean_ioi < 0.25 and ioi_cv < 0.6:
            return RhyContourToken.FLUID
        
        # Syncopated: many notes off-beat
        if on_beat_ratio < 0.2:
            return RhyContourToken.SYNCOPATED
        
        # Default: regular
        return RhyContourToken.REGULAR

    def _extract_dyn(self, midi: pm.PrettyMIDI, start: float, end: float, vel_min: float, vel_max: float):
        velocities = [
            note.velocity
            for inst in midi.instruments
            for note in inst.notes
            if start <= note.start < end
        ]
        
        if not velocities:
            return DynToken.PP
        
        avg = np.mean(velocities)
        normalized = (avg - vel_min) / (vel_max - vel_min)
        normalized = np.clip(normalized, 0.0, 1.0)

        
        # Map to 6 buckets:
        buckets = [DynToken.PP, DynToken.P, DynToken.MP, DynToken.MF, DynToken.F, DynToken.FF]
        idx = min(int(normalized * 6), 5)
        return buckets[idx]

    def _extract_pos(self, idx: int, total: int) -> Token:
        ratio = idx / max(total - 1, 1)
        bucket = round(ratio * 20)
        return PosToken[f"POS_{bucket}"]

    def _get_minmax_vel(self, midi: pm.PrettyMIDI) -> tuple[float, float]:
        """Returns (min, max) velocity over the entire piece, using 5%-95% percentiles
        to be robust against outliers (a single fortissimo accent or an isolated ppp note)."""
        velocities = [
            n.velocity
            for inst in midi.instruments
            for n in inst.notes
        ]
        if not velocities:
            return 0.0, 127.0
        vel_min = float(np.percentile(velocities, 5))
        vel_max = float(np.percentile(velocities, 95))
        if vel_max - vel_min < 1e-6:
            vel_max = vel_min + 1.0
        return vel_min, vel_max
    
    def tokenize(self, path: str) -> np.ndarray:
        midi = self._load(path)
        measures = self._get_measures(midi)
        total = len(measures)
        vel_min, vel_max = self._get_minmax_vel(midi)

        rows = []
        for idx, (start, end) in enumerate(measures):
            harm = self._extract_harm(midi, start, end)
            measure = MacroMeasure(
                harm_main  = harm[0],
                harm_accent= harm[1],
                dens       = self._extract_dens(midi, start, end),
                rhycontour = self._extract_rhycontour(midi, start, end),
                dyn        = self._extract_dyn(midi, start, end, vel_min, vel_max),
                pos        = self._extract_pos(idx, total),
            )
            rows.append(measure.tokenize())
    
        return np.stack(rows)