"""Guitar shapes: a curated table of open chords, plus movable barre shapes for
everything else. Frets are low E to high e; None = muted string, 0 = open."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from chordsheet.guitar.chords import Symbol, parse

# Curated shapes: what a learner should actually play. Values are fret strings,
# one character per string (x = muted).
SHAPES: dict[str, str] = {
    # C
    "C": "x32010", "C7": "x32310", "Cmaj7": "x32000", "Csus2": "x30010", "Csus4": "x33010",
    "C/G": "332010", "C/E": "032010", "C/B": "x22010",
    # D
    "D": "xx0232", "Dm": "xx0231", "D7": "xx0212", "Dm7": "xx0211", "Dmaj7": "xx0222",
    "Dsus4": "xx0233", "Dsus2": "xx0230", "D/F#": "200232", "D/A": "x00232",
    # E
    "E": "022100", "Em": "022000", "E7": "020100", "Em7": "022030", "Emaj7": "021100",
    "Esus4": "022200", "Em/D": "xx0000",
    # F: the four-string "small F" most players learn first; full barre F is
    # the generated E-shape. Fmaj7 is the easiest of all.
    "F": "xx3211", "Fmaj7": "xx3210", "F/C": "x33211", "F/A": "x03211",
    # G
    "G": "320003", "G7": "320001", "Gmaj7": "320002", "Gsus4": "330013", "G/B": "x20003",
    "G/D": "xx0003",
    # A
    "A": "x02220", "Am": "x02210", "A7": "x02020", "Am7": "x02010", "Amaj7": "x02120",
    "Asus4": "x02230", "Asus2": "x02200", "Am/G": "3x2210", "Am/E": "002210",
    # B
    "B7": "x21202", "Bm": "x24432", "Bm7": "x20202",
}
# Shapes with a full barre: playable, but not what the capo search is looking for.
BARRE = {"F/C", "Bm"}

# Movable barre shapes, as fret offsets from the barre fret (None = muted).
_E_SHAPE = {
    "": (0, 2, 2, 1, 0, 0), "m": (0, 2, 2, 0, 0, 0), "7": (0, 2, 0, 1, 0, 0),
    "m7": (0, 2, 0, 0, 0, 0), "maj7": (0, None, 1, 1, 0, None), "sus4": (0, 2, 2, 2, 0, 0),
}
_A_SHAPE = {
    "": (None, 0, 2, 2, 2, 0), "m": (None, 0, 2, 2, 1, 0), "7": (None, 0, 2, 0, 2, 0),
    "m7": (None, 0, 2, 0, 1, 0), "maj7": (None, 0, 2, 1, 2, 0), "sus4": (None, 0, 2, 2, 3, 0),
    "sus2": (None, 0, 2, 2, 0, 0), "dim": (None, 0, 1, 2, 1, None),
}


@dataclass(frozen=True)
class Voicing:
    name: str
    frets: tuple[int | None, ...]  # low E .. high e
    barre: bool = False  # needs a barre (or is a generated barre shape)
    generated: bool = False  # not from the curated table: check it by ear

    @property
    def is_open(self) -> bool:
        return not self.barre

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_frets(s: str) -> tuple[int | None, ...]:
    return tuple(None if ch == "x" else int(ch) for ch in s)


def _barre(sym: Symbol, name: str) -> Voicing | None:
    options = []
    for shape, string_root in ((_E_SHAPE, 4), (_A_SHAPE, 9)):
        if sym.suffix in shape:
            fret = (sym.root - string_root) % 12 or 12
            frets = tuple(None if o is None else fret + o for o in shape[sym.suffix])
            options.append((fret, frets))
    if not options:
        return None
    _, frets = min(options)  # lowest position is easiest
    return Voicing(name, frets, barre=True, generated=True)


def voicing(name: str) -> Voicing | None:
    """Shape for a chord name: curated if we have it, else a movable barre shape.
    A slash chord without its own entry falls back to the plain chord's shape."""
    sym = parse(name)
    if sym is None:
        return None
    for candidate in (sym.name, sym.base.name):
        if candidate in SHAPES:
            return Voicing(name, _parse_frets(SHAPES[candidate]), barre=candidate in BARRE)
    return _barre(sym.base, name)


def is_open_shape(name: str) -> bool:
    v = voicing(name)
    return v is not None and v.is_open
