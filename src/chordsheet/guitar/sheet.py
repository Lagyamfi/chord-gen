"""Layer 4: turn bars of sounding chords into something to play: capo + shapes."""

from __future__ import annotations

from collections import Counter

from chordsheet.guitar.capo import suggest_capo
from chordsheet.guitar.chords import transpose_name
from chordsheet.guitar.voicings import voicing


def guitarify(bars: list[list[dict]], capo: int | None = None) -> dict:
    """Adds a "shape" (what to fret) to every beat dict in place and returns the
    sheet fields for the capo and the voicings used. `capo` overrides the search."""
    durations: Counter[str] = Counter()
    for bar in bars:
        for b in bar:
            durations[b["chord"]] += b["end"] - b["start"]
    choice = suggest_capo(durations)
    fret = choice.fret if capo is None else capo

    shapes: dict[str, dict] = {}
    for bar in bars:
        for b in bar:
            b["shape"] = transpose_name(b["chord"], -fret)
            if b["shape"] not in shapes:
                v = voicing(b["shape"])
                if v is not None:
                    shapes[b["shape"]] = v.to_dict()
    return {
        "capo": {
            "fret": fret,
            "suggested": choice.fret,
            "open_fraction": round(choice.scores[fret], 3),
            "open_fraction_no_capo": round(choice.open_fraction_no_capo, 3),
        },
        "voicings": shapes,
    }
