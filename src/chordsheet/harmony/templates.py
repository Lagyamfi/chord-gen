"""Chord templates and per-beat scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# quality -> (intervals from root, chord-symbol suffix, mir_eval shorthand)
QUALITIES: dict[str, tuple[tuple[int, ...], str, str]] = {
    "maj": ((0, 4, 7), "", "maj"),
    "min": ((0, 3, 7), "m", "min"),
    "7": ((0, 4, 7, 10), "7", "7"),
    "maj7": ((0, 4, 7, 11), "maj7", "maj7"),
    "min7": ((0, 3, 7, 10), "m7", "min7"),
    "sus4": ((0, 5, 7), "sus4", "sus4"),
    "dim": ((0, 3, 6), "dim", "dim"),
}
TRIADS = ("maj", "min")


@dataclass(frozen=True)
class Chord:
    root: int  # pitch class, -1 for no chord
    quality: str  # key of QUALITIES, or "N"

    @property
    def is_none(self) -> bool:
        return self.quality == "N"

    @property
    def intervals(self) -> tuple[int, ...]:
        return QUALITIES[self.quality][0] if not self.is_none else ()

    @property
    def pcs(self) -> set[int]:
        return {(self.root + i) % 12 for i in self.intervals}

    def name(self, bass_pc: int = -1) -> str:
        if self.is_none:
            return "N.C."
        s = NOTE_NAMES[self.root] + QUALITIES[self.quality][1]
        if bass_pc >= 0 and bass_pc != self.root and bass_pc in self.pcs:
            s += "/" + NOTE_NAMES[bass_pc]
        return s

    def mir_eval(self) -> str:
        return "N" if self.is_none else f"{NOTE_NAMES[self.root]}:{QUALITIES[self.quality][2]}"


NO_CHORD = Chord(-1, "N")


def build_templates(qualities=tuple(QUALITIES)) -> tuple[list[Chord], np.ndarray]:
    """Return (chords, unit-norm template matrix). The last entry is no-chord
    and has a zero template; its score is handled in `score_beats`."""
    chords, rows = [], []
    for q in qualities:
        for root in range(12):
            chord = Chord(root, q)
            v = np.zeros(12)
            v[list(chord.pcs)] = 1.0
            chords.append(chord)
            rows.append(v / np.linalg.norm(v))
    chords.append(NO_CHORD)
    rows.append(np.zeros(12))
    return chords, np.array(rows)


def score_beats(
    profiles: np.ndarray,
    bass_pc: np.ndarray,
    chords: list[Chord],
    templates: np.ndarray,
    root_bonus: float = 0.15,
    third_fifth_bonus: float = 0.08,
    no_chord_score: float = 0.2,
) -> np.ndarray:
    """Cosine similarity of each beat profile with each template, plus bass bonuses.
    Silent beats score 1.0 for no-chord. Returns (n_beats, n_chords)."""
    scores = profiles @ templates.T  # profiles and templates are unit-norm
    silent = ~profiles.any(axis=1)
    for j, chord in enumerate(chords):
        if chord.is_none:
            scores[:, j] = np.where(silent, 1.0, no_chord_score)
            continue
        third_fifth = {(chord.root + i) % 12 for i in chord.intervals[1:3]}
        scores[:, j] += np.where(bass_pc == chord.root, root_bonus, 0.0)
        scores[:, j] += np.where(np.isin(bass_pc, list(third_fifth)), third_fifth_bonus, 0.0)
    return scores
