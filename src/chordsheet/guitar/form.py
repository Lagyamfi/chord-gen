"""Song form: the song as a handful of distinct 4-bar lines plus their running order.

A learner holds a song as "verse is G-Bm-C-Am, chorus is Em-C-G-D", not as 150
bars. We cut the bars into 4-bar lines at the offset that makes the most lines
repeat, group lines that match on most of their beats, and letter the groups
A, B, C in order of first appearance. A line that differs from an earlier one
in about one bar is written as a variant, A' (as musicians do), not a new
letter. Each line is shown once, in its most typical version, which also
outvotes one-off misreadings."""

from __future__ import annotations

from itertools import groupby

LINE_BARS = 4
SLOTS = 4  # chord sampled at each quarter of a bar
SAME_LINE = 0.8  # share of matching slots for two lines to count as the same
VARIANT = 0.7  # ... to count as a variant (about one bar different)
NO_CHORDS = "-"
PRIMES = ["′", "″", "‴"]


def bar_slots(bar: list[dict]) -> tuple[str, ...]:
    """The chord (mir_eval label, no bass) under each quarter of the bar."""
    start, end = bar[0]["start"], bar[-1]["end"]
    out = []
    for k in range(SLOTS):
        mid = start + (k + 0.5) * (end - start) / SLOTS
        beat = next((b for b in bar if b["start"] <= mid < b["end"]), bar[-1])
        out.append(beat["mir_eval"])
    return tuple(out)


def similarity(a: list[tuple[str, ...]], b: list[tuple[str, ...]]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    pairs = [(x, y) for ba, bb in zip(a, b) for x, y in zip(ba, bb)]
    return sum(x == y for x, y in pairs) / len(pairs)


def _is_empty(slots: list[tuple[str, ...]]) -> bool:
    return all(x == "N" for bar in slots for x in bar)


def _cut(n_bars: int, offset: int) -> list[tuple[int, int]]:
    """(start, length) of each line: a short lead-in, then 4-bar lines, then any tail."""
    spans = [(0, offset)] if offset else []
    s = offset
    while s < n_bars:
        spans.append((s, min(LINE_BARS, n_bars - s)))
        s += LINE_BARS
    return spans


def song_form(bars: list[list[dict]]) -> dict:
    """{"lines": [{label, start, length}] in song order,
    "letters": [{label, rep, length, count, variant_of}] (rep = bar index of the
               typical version; variant_of = the parent letter, or None),
    "runs": [{label, count, start, end}] the form line with repeats folded}."""
    if not bars:
        return {"lines": [], "letters": [], "runs": []}
    slots = [bar_slots(b) for b in bars]

    def lines_for(offset):
        return [(s, n, slots[s : s + n]) for s, n in _cut(len(bars), offset)]

    def repeat_score(offset):
        full = [
            l for l in lines_for(offset) if l[1] == LINE_BARS and not _is_empty(l[2])
        ]
        return sum(
            max((similarity(a[2], b[2]) for b in full if b is not a), default=0)
            for a in full
        )

    offset = max(range(LINE_BARS), key=lambda o: (round(repeat_score(o), 6), -o))
    lines = lines_for(offset)

    # group: a line joins the first group holding a line it matches closely enough
    groups: list[list[int]] = []
    for i, (_, n, sl) in enumerate(lines):
        if _is_empty(sl):
            continue
        for g in groups:
            if any(similarity(sl, lines[j][2]) >= SAME_LINE for j in g):
                g.append(i)
                break
        else:
            groups.append([i])

    # typical version of each group: the member closest to all the others
    reps = [
        max(g, key=lambda i: sum(similarity(lines[i][2], lines[j][2]) for j in g))
        for g in groups
    ]

    # letters in order of first appearance; a group close to an earlier main
    # letter (about one bar different) becomes its variant: A', A''
    label_of: dict[int, str] = {}
    letters = []
    mains: list[tuple[str, int]] = []  # (label, rep line index)
    for g, rep in zip(groups, reps):
        parent = next(
            (
                lab
                for lab, r in mains
                if similarity(lines[rep][2], lines[r][2]) >= VARIANT
            ),
            None,
        )
        if parent is not None:
            n_variants = sum(1 for l in letters if l["variant_of"] == parent)
            label = (
                parent
                + PRIMES[min(n_variants, len(PRIMES) - 1)]
                + ("" if n_variants < len(PRIMES) else str(n_variants))
            )
        else:
            k = len(mains)
            label = chr(ord("A") + k) if k < 26 else f"Z{k - 25}"
            mains.append((label, rep))
        for i in g:
            label_of[i] = label
        letters.append(
            {
                "label": label,
                "rep": lines[rep][0],
                "length": lines[rep][1],
                "count": len(g),
                "variant_of": parent,
            }
        )

    out_lines = [
        {"label": label_of.get(i, NO_CHORDS), "start": s, "length": n}
        for i, (s, n, _) in enumerate(lines)
    ]
    runs = []
    for label, items in groupby(out_lines, key=lambda l: l["label"]):
        items = list(items)
        runs.append(
            {
                "label": label,
                "count": len(items),
                "start": items[0]["start"],
                "end": items[-1]["start"] + items[-1]["length"] - 1,
            }
        )
    return {"lines": out_lines, "letters": letters, "runs": runs}


def form_text(form: dict) -> str:
    """'A ×4 · B ×2 · - · C ×3'"""
    return " · ".join(
        r["label"] + (f" ×{r['count']}" if r["count"] > 1 else "") for r in form["runs"]
    )
