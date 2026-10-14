"""Parse a pasted chords-over-lyrics chart (Ultimate Guitar style) into the
song's chord sequence, in concert pitch, as mir_eval labels."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple

from chordsheet.harmony.templates import NOTE_NAMES

_ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# chart suffix -> mir_eval quality, longest first so the regex prefers them
_SUFFIXES: dict[str, str] = {
    "maj7": "maj7", "maj9": "maj9", "M7": "maj7", "m7b5": "hdim7", "min7": "min7", "m7": "min7",
    "m9": "min9", "m6": "min6", "madd9": "min(9)", "madd2": "min(9)", "min": "min", "m": "min",
    "dim7": "dim7", "dim": "dim", "°": "dim", "aug": "aug", "+": "aug",
    "7sus4": "sus4(b7)", "7sus": "sus4(b7)", "sus4": "sus4", "sus2": "sus2", "sus": "sus4",
    "add9": "maj(9)", "add2": "maj(9)", "add4": "maj(4)", "13": "13", "11": "11", "9": "9",
    "7": "7", "6": "maj6", "5": "5", "4": "sus4", "2": "sus2", "": "maj",
}
_CHORD_RE = re.compile(
    r"^([A-G])([#b]?)(" + "|".join(re.escape(s) for s in sorted(_SUFFIXES, key=len, reverse=True) if s) + r")?"
    r"(?:\((?:add)?(\d+)\))?(?:/([A-G])([#b]?))?$"
)
_INTERVALS = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]
_IGNORED = re.compile(r"^(\|+|-+|/+|\.+|%|x\d+|\d+x|\(x?\d+x?\)|\*+)$", re.I)
_REPEAT = re.compile(r"(?:^|\s)\(?x\s?(\d+)\)?|\(?(\d+)\s?x\)?(?:\s|$)", re.I)
_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$")
_ANCHOR = re.compile(r"@\s*(\d+):(\d+(?:\.\d+)?)")
_CAPO = re.compile(r"capo\s*(?:on|:|-)?\s*(\d+)", re.I)


@dataclass(frozen=True)
class ChartChord:
    root: int
    quality: str  # mir_eval quality
    bass: int = 0  # bass interval above root, in semitones

    @property
    def is_none(self) -> bool:
        return self.quality == "N"

    def transpose(self, semitones: int) -> ChartChord:
        if self.is_none:
            return self
        return ChartChord((self.root + semitones) % 12, self.quality, self.bass)

    @property
    def label(self) -> str:
        if self.is_none:
            return "N"
        s = f"{NOTE_NAMES[self.root]}:{self.quality}"
        return s + (f"/{_INTERVALS[self.bass]}" if self.bass else "")


def _pc(letter: str, accidental: str) -> int:
    return (_ROOTS[letter] + {"#": 1, "b": -1}.get(accidental, 0)) % 12


NO_CHORD = ChartChord(-1, "N")
_NC = re.compile(r"^n\.?c\.?$", re.I)


def parse_chord(token: str) -> ChartChord | None:
    if _NC.match(token.strip()):
        return NO_CHORD
    m = _CHORD_RE.match(token.strip())
    if not m:
        return None
    letter, acc, suffix, paren, bass_letter, bass_acc = m.groups()
    root = _pc(letter, acc)
    quality = _SUFFIXES[suffix or ""]
    if paren and quality in ("maj", "min"):
        quality = f"{quality}({paren})"
    bass = (_pc(bass_letter, bass_acc) - root) % 12 if bass_letter else 0
    return ChartChord(root, quality, bass)


def parse_chord_line(line: str) -> list[ChartChord] | None:
    """Chords on a line if every meaningful token is a chord, else None (lyrics)."""
    tokens = [t.strip("|()[]") for t in line.split()]
    tokens = [t for t in tokens if t and not _IGNORED.match(t)]
    if not tokens:
        return None
    chords = [parse_chord(t) for t in tokens]
    if any(c is None for c in chords):
        return None
    m = _REPEAT.search(line)
    return chords * (int(m.group(1) or m.group(2)) if m else 1)


class SeqItem(NamedTuple):
    chord: ChartChord
    section: str
    section_index: int


@dataclass
class Section:
    name: str
    chords: list[ChartChord] = field(default_factory=list)
    repeat: int = 1
    reused: bool = False
    anchor: float | None = None  # "[Verse 1] @1:08": section starts here in the recording  # chords copied from an earlier section of the same kind

    @property
    def kind(self) -> str:
        return re.sub(r"[\s\d]+$", "", self.name.lower()).strip()


@dataclass
class Chart:
    sections: list[Section]
    capo: int = 0

    def anchors(self) -> list[tuple[int, float, str]]:
        """(sequence position of the section's first chord, time, section name)."""
        seq = self.sequence()
        out = []
        for i, sec in enumerate(self.sections):
            if sec.anchor is None:
                continue
            pos = next((j for j, it in enumerate(seq) if it.section_index == i), None)
            if pos is None:
                raise ValueError(f"[{sec.name}] @anchor: its first chord merged into the previous section")
            out.append((pos, sec.anchor, sec.name))
        return out

    def sequence(self) -> list[SeqItem]:
        """Concert-pitch chord changes in song order; consecutive duplicates merged
        (a merged chord belongs to the earlier section), except at the start of an
        anchored section, which always gets its own position to anchor to."""
        seq: list[SeqItem] = []
        for i, sec in enumerate(self.sections):
            for j, chord in enumerate(sec.chords * sec.repeat):
                chord = chord.transpose(self.capo)
                if not seq or seq[-1].chord != chord or (j == 0 and sec.anchor is not None):
                    seq.append(SeqItem(chord, sec.name, i))
        return seq


def parse_chart(text: str) -> Chart:
    capo_m = _CAPO.search(text)
    sections: list[Section] = []
    current = Section("Start")
    for line in text.splitlines():
        sec_m = _SECTION.match(line)
        if sec_m:
            sections.append(current)
            current = Section(sec_m.group(1).strip())
            rest = sec_m.group(2)
            anchor = _ANCHOR.search(rest)
            if anchor:
                current.anchor = int(anchor.group(1)) * 60 + float(anchor.group(2))
                rest = _ANCHOR.sub("", rest)
            rep = _REPEAT.search(rest)
            if rep and parse_chord_line(rest) is None:
                current.repeat = int(rep.group(1) or rep.group(2))
            elif rest.strip():
                current.chords.extend(parse_chord_line(rest) or [])
            continue
        chords = parse_chord_line(line)
        if chords:
            current.chords.extend(chords)
    sections.append(current)

    # A section with no chords repeats the last section of the same kind
    # ("[Chorus]" written once in full, later only as a heading or lyrics).
    out: list[Section] = []
    for sec in sections:
        if not sec.chords:
            prior = next((p for p in reversed(out) if p.kind == sec.kind and p.chords), None)
            if prior is None:
                continue
            sec.chords, sec.reused = list(prior.chords), True
        out.append(sec)
    return Chart(out, int(capo_m.group(1)) if capo_m else 0)
