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


def _header(sheet: dict) -> str:
    capo = sheet.get("capo", {}).get("fret", 0)
    capo_text = f"Capo {capo} (play the shapes row)" if capo else "No capo"
    key_note = " (set)" if sheet.get("key_source") == "set" else ""
    return f"Key: {sheet['key']}{key_note} · {capo_text} · {sheet['tempo']:.0f} BPM · transcriber: {sheet['transcriber']}"


def render_song(sheet: dict, width: int = 10) -> str:
    """One-page chart: the form line, then each distinct 4-bar line once."""
    from chordsheet.guitar.form import form_text

    form = sheet.get("form")
    if not form or not form["letters"]:
        return render_chart(sheet, width=width)
    capo = sheet.get("capo", {}).get("fret", 0)
    from chordsheet.guitar.sections import sections_for

    sections = sheet.get("sections") or sections_for(sheet)
    rep_of = {L["label"]: L for L in form["letters"]}
    lines = [_header(sheet), ""]
    if sections["runs"]:
        lines += ["Form: " + " · ".join(r["name"] + (f" ×{r['reps']}" if r["reps"] > 1 else "") for r in sections["runs"]), ""]
        for block in sections["blocks"]:
            lines.append(f"[{block['title']}]" + (f"  ×{block['count']}" if block["count"] > 1 else ""))
            for label, times in block["lines"]:
                L = rep_of[label]
                bars = sheet["bars"][L["rep"] : L["rep"] + L["length"]]
                tag = f"{label:<3}" + (f"×{times}" if times > 1 else "  ")
                pad = " " * len(tag)
                lines.append(f"{tag} | " + " | ".join(_collapse(b, "chord").ljust(width) for b in bars) + " |")
                lines.append(f"{pad} | " + " | ".join(_collapse(b, "roman").ljust(width) for b in bars) + " |")
                if capo:
                    lines.append(f"{pad} | " + " | ".join(_collapse(b, "shape").ljust(width) for b in bars) + " |  <- play")
            lines.append("")
    else:
        lines += ["Form: " + form_text(form), ""]
    return "\n".join(lines + _footer(sheet))


def _footer(sheet: dict) -> list[str]:
    out = []
    outside = sorted({b["chord"] for bar in sheet["bars"] for b in bar if not b["in_key"]})
    if outside:
        out.append("Outside the key (worth learning): " + ", ".join(outside))
    shapes = sheet.get("voicings", {})
    if shapes:
        out.append("")
        out.append("Shapes: " + "  ".join(f"{n} {''.join('x' if f is None else str(f) for f in v['frets'])}"
                                       + (" (barre)" if v["barre"] else "") for n, v in shapes.items()))
    return out


def render_chart(sheet: dict, bars_per_line: int = 4, width: int = 10) -> str:
    capo = sheet.get("capo", {}).get("fret", 0)
    lines = [_header(sheet), ""]
    bars = sheet["bars"]
    for i in range(0, len(bars), bars_per_line):
        row = bars[i : i + bars_per_line]
        lines.append("| " + " | ".join(_collapse(b, "chord").ljust(width) for b in row) + " |")
        lines.append("| " + " | ".join(_collapse(b, "roman").ljust(width) for b in row) + " |")
        if capo:
            lines.append("| " + " | ".join(_collapse(b, "shape").ljust(width) for b in row) + " |  <- play")
        lines.append("")
    return "\n".join(lines + _footer(sheet))
