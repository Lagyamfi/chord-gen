"""Minimal text chord chart. Capo search and voicings land in week 3."""

from __future__ import annotations


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
    out: list[str] = []
    for b in bar:
        if not out or out[-1] != b[field]:
            out.append(b[field])
    return " ".join(out)


def render_chart(sheet: dict, bars_per_line: int = 4, width: int = 10) -> str:
    header = f"Key: {sheet['key']} · {sheet['tempo']:.0f} BPM · transcriber: {sheet['transcriber']}"
    lines = [header, ""]
    bars = sheet["bars"]
    for i in range(0, len(bars), bars_per_line):
        row = bars[i : i + bars_per_line]
        lines.append("| " + " | ".join(_collapse(b, "chord").ljust(width) for b in row) + " |")
        lines.append("| " + " | ".join(_collapse(b, "roman").ljust(width) for b in row) + " |")
        lines.append("")
    outside = sorted({b["chord"] for bar in bars for b in bar if not b["in_key"]})
    if outside:
        lines.append("Outside the key (worth learning): " + ", ".join(outside))
    return "\n".join(lines)
