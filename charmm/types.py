# charmm/types.py

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


# ── Token base ────────────────────────────────────────────────────────────────

class Token(Enum):
    """Abstract base for all ChARMM tokens."""
    pass


# ── Macro tokens ──────────────────────────────────────────────────────────────

_ROOTS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_MODES = ["maj", "min", "dim", "amb"]
HarmToken = Enum("HarmToken", {
    f"{r}_{m}": auto()
    for r in _ROOTS
    for m in _MODES
}, type=Token)


class DensToken(Token):
    SPARSE = auto()
    MEDIUM = auto()
    DENSE  = auto()


class RhyContourToken(Token):
    REGULAR    = auto()
    SYNCOPATED = auto()
    FLUID      = auto()


class DynToken(Token):
    PP = auto()
    P  = auto()
    MP = auto()
    MF = auto()
    F  = auto()
    FF = auto()


PosToken = Enum("PosToken", {
    f"POS_{i}": auto()
    for i in range(21)
}, type=Token)


# ── Vocab ─────────────────────────────────────────────────────────────────────

class _SubVocab:
    """Independent vocabulary for a single token dimension. IDs start from 0."""

    def __init__(self, token_cls: type[Token]) -> None:
        self._token_to_id: dict[Token, int] = {}
        self._id_to_token: dict[int, Token] = {}
        for idx, member in enumerate(token_cls):
            self._token_to_id[member] = idx
            self._id_to_token[idx] = member

    def encode(self, token: Token) -> int:
        return self._token_to_id[token]

    def decode(self, idx: int) -> Token:
        return self._id_to_token[idx]

    def __len__(self) -> int:
        return len(self._token_to_id)


class Vocab:
    """
    Multi-stream token registry.
    Each macro dimension has its own independent vocabulary.

    Usage:
        vocab.harm.encode(HarmToken.C_min)  → int
        vocab.harm.decode(3)                → HarmToken.C_min
        vocab.len_harm                      → 48
    """

    _instance: ClassVar[Vocab | None] = None

    def __new__(cls) -> Vocab:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._build()
        return cls._instance

    def _build(self) -> None:
        self.harm = _SubVocab(HarmToken)      # type: ignore[arg-type]
        self.dens = _SubVocab(DensToken)
        self.rhy  = _SubVocab(RhyContourToken)
        self.dyn  = _SubVocab(DynToken)
        self.pos  = _SubVocab(PosToken)       # type: ignore[arg-type]

    @property
    def len_harm(self) -> int: return len(self.harm)

    @property
    def len_dens(self) -> int: return len(self.dens)

    @property
    def len_rhy(self) -> int: return len(self.rhy)

    @property
    def len_dyn(self) -> int: return len(self.dyn)

    @property
    def len_pos(self) -> int: return len(self.pos)


# ── Module-level singleton ────────────────────────────────────────────────────

vocab = Vocab()


# ── MacroMeasure ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MacroMeasure:
    harm:       HarmToken       # type: ignore[valid-type]
    dens:       DensToken
    rhycontour: RhyContourToken
    dyn:        DynToken
    pos:        PosToken        # type: ignore[valid-type]

    def tokenize(self) -> np.ndarray:
        """Returns a (5,) int32 array of encoded token IDs."""
        return np.array([
            vocab.harm.encode(self.harm),
            vocab.dens.encode(self.dens),
            vocab.rhy.encode(self.rhycontour),
            vocab.dyn.encode(self.dyn),
            vocab.pos.encode(self.pos),
        ], dtype=np.int32)