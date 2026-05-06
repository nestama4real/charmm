import pretty_midi as pm
import numpy as np
from charmm.types import *
class MacroTokenizer:
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
        pass

    def _extract_dens(self, midi: pm.PrettyMIDI, start: float, end: float) -> DensToken:
        pass

    def _extract_rhycontour(self, midi: pm.PrettyMIDI, start: float, end: float):
        pass

    def _extract_dyn(self, midi: pm.PrettyMIDI, start: float, end: float):
        pass

    def _extract_pos(self, idx: int, total: int) -> Token:
        ratio = idx / max(total - 1, 1)
        bucket = round(ratio * 20)
        return PosToken[f"POS_{bucket}"]

    def tokenize(self, path: str) -> np.ndarray:
        midi = self._load(path)
        measures = self._get_measures(midi)
        total = len(measures)

        rows = []
        for idx, (start, end) in enumerate(measures):
            measure = MacroMeasure(
                harm       = self._extract_harm(midi, start, end),
                dens       = self._extract_dens(midi, start, end),
                rhycontour = self._extract_rhycontour(midi, start, end),
                dyn        = self._extract_dyn(midi, start, end),
                pos        = self._extract_pos(idx, total),
            )
            rows.append(measure.tokenize())
    
        return np.stack(rows)