import numpy as np
import pretty_midi as pm
from charmm.types import *

class HarmonicExtractor:
    KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    DIM_TEMPLATE    = np.array([5.5,  1.5,  2.0,  5.0,  1.5,  2.5,  5.0,  1.5,  3.5,  4.5,  2.0,  1.5 ])

    ROOTS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    # Thresholds (calibrate empirically)
    AMB_THRESHOLD     = 0.5   # below this correlation → ambiguous
    ACCENT_THRESHOLD  = 0.4   # accent must be at least this convincing
    RESIDUAL_MIN      = 0.15  # residual must contain at least 15% of original mass

    def extract(self, midi: pm.PrettyMIDI, start: float, end: float) -> tuple[Token, Token]:
        profile = self._build_profile(midi, start, end)
        
        if profile.sum() == 0:
            return HarmToken.C_amb, HarmToken.NONE
        
        # Pass 1: dominant color
        main_token, _ = self._detect_color(profile)
        
        # Pass 2: residual analysis
        residual = self._subtract_contribution(profile, main_token)
        if residual.sum() < self.RESIDUAL_MIN * profile.sum():
            return main_token, HarmToken.NONE
        
        accent_token, accent_score = self._detect_color(residual)
        
        # Reject accent if too weak or same root as main
        main_root = main_token.name.split("_")[0]
        accent_root = accent_token.name.split("_")[0]
        if accent_score < self.ACCENT_THRESHOLD or accent_root == main_root:
            return main_token, HarmToken.NONE
        
        return main_token, accent_token
    
    def _build_profile(self, midi: pm.PrettyMIDI, start: float, end: float) -> np.ndarray:
        """
        Weighted pitch class profile (12,).
        Weights: duration + downbeat bonus + bass bonus.
        """
        profile = np.zeros(12)
        notes_in_measure = [
            n for inst in midi.instruments for n in inst.notes
            if start <= n.start < end
        ]
        if not notes_in_measure:
            return profile

        bass_pitch = min(n.pitch for n in notes_in_measure)
        measure_duration = end - start
        
        for n in notes_in_measure:
            pc = n.pitch % 12
            weight = n.end - n.start
            # Downbeat bonus: notes starting near the beginning of the measure
            if (n.start - start) / measure_duration < 0.1:
                weight *= 1.5
            # Bass bonus: lowest pitch class gets extra weight
            if n.pitch == bass_pitch:
                weight *= 1.3
            profile[pc] += weight
        
        return profile

    def _detect_color(self, profile: np.ndarray) -> tuple[Token, float]:
        """
        Returns (best Token, correlation score).
        Tries all 48 maj/min/dim candidates + amb fallback.
        """
        if profile.sum() == 0:
            return HarmToken.C_amb, 0.0
        
        best_score = -np.inf
        best_token = HarmToken.C_amb
        
        for root in range(12):
            candidates = [
                (np.roll(self.KRUMHANSL_MAJOR, root), "maj"),
                (np.roll(self.KRUMHANSL_MINOR, root), "min"),
                (np.roll(self.DIM_TEMPLATE,    root), "dim"),
            ]
            for template, mode in candidates:
                score = np.corrcoef(profile, template)[0, 1]
                if np.isnan(score):
                    continue
                if score > best_score:
                    best_score = score
                    best_token = HarmToken[f"{self.ROOTS[root]}_{mode}"]
        
        n_active = (profile > 0).sum()
        threshold = self.AMB_THRESHOLD if n_active >= 4 else 0.3
        # If correlation too weak → ambiguous color, keep the dominant root
        if best_score < threshold:
            dom_root = int(np.argmax(profile))
            return HarmToken[f"{self.ROOTS[dom_root]}_amb"], best_score
        
        return best_token, best_score


    def _subtract_contribution(self, profile: np.ndarray, token: Token) -> np.ndarray:
        """
        Remove the contribution of `token` from the profile.
        Strategy: zero out (or strongly attenuate) the pitch classes that belong
        to this token's chord, weighted by their template strength.
        """
        if token.name.endswith("_amb") or token.name == "NONE":
            return np.zeros_like(profile)  # nothing meaningful left to analyse
        
        root = self.ROOTS.index(token.name.split("_")[0])
        mode = token.name.split("_")[1]
        
        if mode == "maj":   template = np.roll(self.KRUMHANSL_MAJOR, root)
        elif mode == "min": template = np.roll(self.KRUMHANSL_MINOR, root)
        else:               template = np.roll(self.DIM_TEMPLATE, root)
        
        # Normalize template to a contribution mask in [0, 1]
        mask = template / template.max()
        residual = profile * (1 - mask)
        return np.maximum(residual, 0)

