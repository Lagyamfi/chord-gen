"""Text chord chart: sounding chords, Roman numerals and, with a capo, the shapes to play."""

from __future__ import annotations

from collections import Counter


def group_bars(beats: list[dict], downbeats: list[float]) -> list[list[dict]]:
    """Split beat dicts into bars at the downbeat times."""
    starts = set(round(d, 3) for d in downbeats)
    bars: list[list[dict]] = []
    for b in beats:
        if not bars or round(b["start"], 3) in starts:
            bars.append([])
        bars[-1].append(b)
    return bars


def _collapse(bar: list[dict], field: str) -> str:
    """One entry per chord in the bar. Beats are grouped by chord ignoring the bass
    (a walking bass would otherwise print "C C/E C"); a slash name is shown only
    when that bass is the most common one over the chord."""
    groups: list[list[str]] = []
    keys: list[str] = []
    for b in bar:
        key = b["mir_eval"]
        if not keys or keys[-1] != key:
            keys.append(key)
            groups.append([])
        groups[-1].append(b[field])
    out = []
    for names in groups:
        counts = Counter(names)
        plain = next((n for n in names if "/" not in n), names[0].split("/")[0])
        out.append(max(counts, key=lambda n: (counts[n], n == plain)))
    return " ".join(out)


def render_chart(sheet: dict, bars_per_line: int = 4, width: int = 10) -> str:
    capo = sheet.get("capo", {}).get("fret", 0)
    capo_text = f"Capo {capo} (play the shapes row)" if capo else "No capo"
    key_note = " (set)" if sheet.get("key_source") == "set" else ""
    header = f"Key: {sheet['key']}{key_note} · {capo_text} · {sheet['tempo']:.0f} BPM · transcriber: {sheet['transcriber']}"
    lines = [header, ""]
    bars = sheet["bars"]
    for i in range(0, len(bars), bars_per_line):
        row = bars[i : i + bars_per_line]
        lines.append("| " + " | ".join(_collapse(b, "chord").ljust(width) for b in row) + " |")
        lines.append("| " + " | ".join(_collapse(b, "roman").ljust(width) for b in row) + " |")
        if capo:
            lines.append("| " + " | ".join(_collapse(b, "shape").ljust(width) for b in row) + " |  <- play")
        lines.append("")
    outside = sorted({b["chord"] for bar in bars for b in bar if not b["in_key"]})
    if outside:
        lines.append("Outside the key (worth learning): " + ", ".join(outside))
    shapes = sheet.get("voicings", {})
    if shapes:
        lines.append("")
        lines.append("Shapes: " + "  ".join(f"{n} {''.join('x' if f is None else str(f) for f in v['frets'])}"
                                         + (" (barre)" if v["barre"] else "") for n, v in shapes.items()))
    return "\n".join(lines)
