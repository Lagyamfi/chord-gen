"""Capo search: the capo position that puts the most playing time on open shapes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from chordsheet.guitar.chords import parse, transpose_name
from chordsheet.guitar.voicings import is_open_shape

MAX_CAPO = 7


@dataclass(frozen=True)
class CapoChoice:
    fret: int
    open_fraction: float  # share of chord time on open shapes with this capo
    open_fraction_no_capo: float
    scores: dict[int, float]


def suggest_capo(durations: Counter[str], per_fret: float = 0.03) -> CapoChoice:
    """`durations`: chord name -> seconds (or beats) it is played. With capo c the
    hands play every chord c semitones lower. Maximises the open-shape fraction
    minus `per_fret` per capo fret: a capo changes the sound, so it has to earn
    its place (capo 5 for 20% more open time is not worth it; capo 3 to turn Bb
    and Gm into G and Em is). Ties go to the lower capo."""
    chords = {n: d for n, d in durations.items() if parse(n) is not None}
    total = sum(chords.values())
    scores = {}
    for c in range(MAX_CAPO + 1):
        on_open = sum(d for n, d in chords.items() if is_open_shape(transpose_name(n, -c)))
        scores[c] = on_open / total if total else 0.0
    fret = max(scores, key=lambda c: (round(scores[c] - per_fret * c, 6), -c))
    return CapoChoice(fret, scores[fret], scores[0], scores)
