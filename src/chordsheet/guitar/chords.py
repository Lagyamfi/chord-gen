"""Chord-symbol parsing and transposition for sheet chord names ("C#m", "G/B", "N.C.")."""

from __future__ import annotations

import re
from dataclasses import dataclass

from chordsheet.harmony.templates import NOTE_NAMES

NO_CHORD = "N.C."
_NAME = re.compile(r"^([A-G])([#b]?)(maj7|m7|m|7|sus4|sus2|dim|)(?:/([A-G])([#b]?))?$")
_ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _pc(letter: str, accidental: str) -> int:
    return (_ROOTS[letter] + {"#": 1, "b": -1}.get(accidental, 0)) % 12


@dataclass(frozen=True)
class Symbol:
    root: int
    suffix: str  # "", "m", "7", "maj7", "m7", "sus4", "sus2", "dim"
    bass: int | None = None  # pitch class of a slash bass

    @property
    def name(self) -> str:
        s = NOTE_NAMES[self.root] + self.suffix
        return s + (f"/{NOTE_NAMES[self.bass]}" if self.bass is not None else "")

    @property
    def base(self) -> Symbol:
        return Symbol(self.root, self.suffix)

    def transpose(self, semitones: int) -> Symbol:
        bass = None if self.bass is None else (self.bass + semitones) % 12
        return Symbol((self.root + semitones) % 12, self.suffix, bass)


def parse(name: str) -> Symbol | None:
    """Sheet chord name -> Symbol; None for N.C. or anything unrecognised."""
    m = _NAME.match(name.strip())
    if not m:
        return None
    letter, acc, suffix, bass_letter, bass_acc = m.groups()
    bass = _pc(bass_letter, bass_acc) if bass_letter else None
    return Symbol(_pc(letter, acc), suffix, bass)


def transpose_name(name: str, semitones: int) -> str:
    sym = parse(name)
    return name if sym is None else sym.transpose(semitones).name
