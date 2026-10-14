"""Key detection (Krumhansl-Kessler) and Roman-numeral analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chordsheet.harmony.templates import NOTE_NAMES, Chord

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
# natural minor plus the raised 7th (harmonic minor), so V and vii° count as diatonic
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10, 11)

# semitones above the tonic -> numeral
NUMERALS = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]
NUMERAL_SUFFIX = {"maj": "", "min": "", "7": "7", "maj7": "maj7", "min7": "7", "sus4": "sus4", "dim": "°"}


@dataclass(frozen=True)
class Key:
    tonic: int
    mode: str  # "major" | "minor"
    score: float

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} {self.mode}"

    @property
    def scale(self) -> set[int]:
        steps = MAJOR_SCALE if self.mode == "major" else MINOR_SCALE
        return {(self.tonic + s) % 12 for s in steps}


def estimate_keys(profiles: np.ndarray) -> list[Key]:
    """All 24 keys ranked by correlation with the summed profile."""
    total = profiles.sum(axis=0)
    keys = []
    for tonic in range(12):
        for mode, ref in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            r = float(np.corrcoef(total, np.roll(ref, tonic))[0, 1]) if total.any() else 0.0
            keys.append(Key(tonic, mode, r))
    return sorted(keys, key=lambda k: -k.score)


def roman(chord: Chord, key: Key) -> str:
    if chord.is_none:
        return "-"
    numeral = NUMERALS[(chord.root - key.tonic) % 12]
    if chord.quality in ("min", "min7", "dim"):
        numeral = numeral.lower()
    return numeral + NUMERAL_SUFFIX[chord.quality]


def in_key(chord: Chord, key: Key) -> bool:
    return chord.is_none or chord.pcs <= key.scale
