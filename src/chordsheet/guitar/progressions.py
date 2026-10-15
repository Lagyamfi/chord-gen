"""Chord progressions: the Roman-numeral loop behind each line, matched across songs.

What transfers between songs is the progression, not the chord names: the
four-chord loop is G-D-Em-C in one song and C-G-Am-F in another. A progression
is treated as a loop, so I-V-vi-IV and vi-IV-I-V (the same loop started on vi)
match, and a line that plays a short loop twice (I-IV-I-IV) reduces to it."""

from __future__ import annotations

import re

# loops a learner meets often, written from their usual starting chord
FAMILIES: dict[str, list[str]] = {
    "Four-chord loop": ["I", "V", "vi", "IV"],
    "Two-chord vamp": ["I", "IV"],
    "Three-chord (I–IV–V)": ["I", "IV", "V"],
    "'50s progression": ["I", "vi", "IV", "V"],
    "ii–V–I": ["ii", "V", "I"],
}

_EXTENSION = re.compile(r"(maj7|7|sus4|sus2|°|dim)$")


def simplify(numeral: str) -> str:
    """'V7' -> 'V', 'IVmaj7' -> 'IV', 'viisus4' -> 'vii': match on the triad."""
    return _EXTENSION.sub("", numeral)


def line_numerals(bars: list[list[dict]]) -> list[str]:
    """The numerals of a stretch of bars in order, repeats collapsed, no-chord skipped."""
    seq: list[str] = []
    for bar in bars:
        for beat in bar:
            r = beat["roman"]
            if r != "-" and (not seq or seq[-1] != r):
                seq.append(r)
    return seq


def as_loop(seq: list[str]) -> list[str]:
    """A line read as a loop: a last chord equal to the first is the loop coming
    round (dropped), and a line that plays a shorter loop twice is reduced to it:
    I-IV-I-IV -> I-IV;  I-vi-V-IV-I -> I-vi-V-IV."""
    s = [simplify(x) for x in seq]
    if len(s) > 2 and s[0] == s[-1]:
        s = s[:-1]
    for p in range(1, len(s)):
        if len(s) % p == 0 and all(s[i] == s[i % p] for i in range(len(s))):
            return s[:p]
    return s


def loop_key(loop: list[str]) -> tuple[str, ...]:
    """The same key for every rotation of a loop: its alphabetically first rotation."""
    if not loop:
        return ()
    return min(tuple(loop[i:] + loop[:i]) for i in range(len(loop)))


_FAMILY_BY_KEY = {loop_key(v): (name, v) for name, v in FAMILIES.items()}


def describe(seq: list[str]) -> dict | None:
    """{numerals, loop, key, name, start} for a line's numerals, or None when the
    line has fewer than two chords. `start` is set when the song starts a named
    loop somewhere other than its usual first chord ("from vi")."""
    loop = as_loop(seq)
    if len(loop) < 2:
        return None
    key = loop_key(loop)
    family = _FAMILY_BY_KEY.get(key)
    start = None
    if family and loop[0] != family[1][0]:
        start = loop[0]
    return {
        "numerals": seq,
        "loop": loop,
        "key": "-".join(key),
        "name": family[0] if family else None,
        "start": start,
    }


def label(desc: dict) -> str:
    """'vi – IV – I – V · Four-chord loop, from vi'"""
    text = " – ".join(desc["loop"])
    if desc["name"]:
        text += f" · {desc['name']}" + (
            f", from {desc['start']}" if desc["start"] else ""
        )
    return text


def section_progressions(sheet: dict) -> list[list[dict]]:
    """For each section block of a sheet, the progressions of its distinct lines
    (in line order, each loop once): [[{label, numerals, loop, key, name, start, time}]]."""
    form, sections = sheet.get("form") or {}, sheet.get("sections") or {}
    letters = {L["label"]: L for L in form.get("letters", [])}
    out = []
    for block in sections.get("blocks", []):
        seen, items = set(), []
        for entry in block["lines"]:
            lab, starts = entry[0], entry[2] if len(entry) > 2 else []
            L = letters.get(lab)
            if not L:
                continue
            desc = describe(
                line_numerals(sheet["bars"][L["rep"] : L["rep"] + L["length"]])
            )
            if desc and desc["key"] not in seen:
                seen.add(desc["key"])
                items.append(
                    {**desc, "line": lab, "time": starts[0] if starts else None}
                )
        out.append(items)
    return out


def library(songs: list[dict]) -> dict[str, list[dict]]:
    """Progression key -> every place it occurs across songs.

    `songs`: [{"id", "title", "sheet"}]. Each occurrence:
    {song, title, section, line, time, name, loop}."""
    lib: dict[str, list[dict]] = {}
    for song in songs:
        blocks = (song["sheet"].get("sections") or {}).get("blocks", [])
        for block, items in zip(blocks, section_progressions(song["sheet"])):
            for p in items:
                lib.setdefault(p["key"], []).append(
                    {
                        "song": song["id"],
                        "title": song["title"],
                        "section": block["title"],
                        "line": p["line"],
                        "time": p["time"],
                        "name": p["name"],
                        "loop": p["loop"],
                    }
                )
    return lib
