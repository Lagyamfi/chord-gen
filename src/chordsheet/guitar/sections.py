"""Guess section names (Intro, Verse, Pre-chorus, Chorus, Bridge, Outro) for each line.

Names go on each occurrence of a line, not on its letter: the same chords often
serve the verse and the chorus. The guess uses what a listener uses besides the
chords: choruses are louder and fuller (more notes), verses quieter, and a song
opens with an intro and closes with an outro. Lawrence can rename any section
in the player; this only has to be a good first draft."""

from __future__ import annotations

from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np

from chordsheet.models import NoteEvent

NAMES = ("Intro", "Verse", "Pre-chorus", "Chorus", "Bridge", "Outro")


def line_energy(
    bars: list[list[dict]], lines: list[dict], audio_path: Path, notes: list[NoteEvent]
) -> np.ndarray:
    """One loudness-and-fullness score per line: RMS level (dB) plus note density,
    each as a z-score within the song, averaged."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    onsets = np.array([n.onset for n in notes if not n.is_drum])
    db, dens = [], []
    for line in lines:
        a = bars[line["start"]][0]["start"]
        b = bars[line["start"] + line["length"] - 1][-1]["end"]
        seg = rms[(t >= a) & (t < b)]
        db.append(20 * np.log10(seg.mean() + 1e-9) if len(seg) else -120.0)
        dens.append(np.sum((onsets >= a) & (onsets < b)) / max(b - a, 1e-6))
    z = lambda v: (np.asarray(v) - np.median(v)) / (np.std(v) + 1e-9)
    return (z(db) + z(np.log1p(dens))) / 2


def guess_sections(lines: list[dict], energy: np.ndarray) -> list[str]:
    """A section name for every line (lines as in song_form()["lines"])."""
    n = len(lines)
    base = [l["label"].rstrip("′″‴") for l in lines]
    names = [""] * n
    playing = [i for i in range(n) if base[i] != "-"]
    if not playing:
        return ["Intro"] * n

    # loud lines are chorus-like: split at the midpoint between the louder and quieter halves
    e = energy[playing]
    hi = e >= (np.percentile(e, 75) + np.percentile(e, 25)) / 2
    loud = {i: bool(h) for i, h in zip(playing, hi)}

    first_loud = next((i for i in playing if loud[i]), playing[0])
    last_loud = max((i for i in playing if loud[i]), default=playing[-1])

    # Intro: anything before the first repeated letter, or quieter than every verse would be
    repeated = {b for b, c in Counter(base[i] for i in playing).items() if c > 1}
    first_body = next((i for i in playing if base[i] in repeated), playing[0])
    quiet_floor = np.percentile(e, 15)
    for i in range(n):
        if i < first_body or (
            i < first_loud and base[i] != "-" and energy[i] < quiet_floor - 0.5
        ):
            names[i] = "Intro"
        elif base[i] == "-":
            names[i] = "Intro" if i < first_loud else ("Outro" if i > last_loud else "")

    # Outro: quiet lines after the last loud stretch
    for i in range(last_loud + 1, n):
        if not names[i]:
            names[i] = "Outro"

    # Chorus: loud lines. Bridge: a loud or quiet stretch late in the song whose
    # letters appear nowhere else
    seen_before = lambda i: any(
        base[j] == base[i]
        for j in range(n)
        if j != i
        and abs(j - i) > 1
        and not (j > i and all(base[k] == base[i] for k in range(i, j)))
    )
    for i in playing:
        if names[i]:
            continue
        unique = not seen_before(i)
        if unique and i > n / 2 and i < last_loud:
            names[i] = "Bridge"
        elif loud[i]:
            names[i] = "Chorus"

    # the quiet lines left are verses, except the last line or two of a stretch
    # that leads into a chorus with letters the verses do not use: pre-chorus
    stretches: list[list[int]] = []
    for i in (i for i in playing if not names[i]):
        if stretches and stretches[-1][-1] == i - 1:
            stretches[-1].append(i)
        else:
            stretches.append([i])
    verse_letters = Counter(base[st[0]] for st in stretches)
    for st in stretches:
        before_chorus = st[-1] + 1 < n and names[st[-1] + 1] == "Chorus"
        for k, i in enumerate(st):
            tail = k >= len(st) - 2 and k > 0
            pre = before_chorus and tail and verse_letters[base[i]] <= 1 and base[i] != base[st[0]]
            names[i] = "Pre-chorus" if pre else "Verse"
    return [nm or "Verse" for nm in names]


def section_runs(lines: list[dict], names: list[str]) -> list[dict]:
    """Consecutive lines with the same name -> [{name, start, end, line_indexes}]."""
    runs = []
    for name, grp in groupby(range(len(lines)), key=lambda i: names[i]):
        idx = list(grp)
        runs.append(
            {
                "name": name,
                "start": lines[idx[0]]["start"],
                "end": lines[idx[-1]]["start"] + lines[idx[-1]]["length"] - 1,
                "lines": idx,
            }
        )
    return runs


PRESETS = ("Intro", "Verse", "Pre-chorus", "Chorus", "Bridge", "Instrumental", "Tag", "Outro")


def _base(label: str) -> str:
    return label.rstrip("′″‴")


def _unit(seq: list[str]) -> tuple[list[list], int]:
    """A section's lines as [[label, times played in a row], ...] plus how many
    times the whole section repeats back to back.

    [E, F, E, F] -> ([[E, 1], [F, 1]], 2): a chorus sung twice ("Chorus x2")
    [A, A, B]    -> ([[A, 2], [B, 1]], 1): a line played twice inside a verse
    A single line repeated ([A, A]) is a line repeat, not a section repeat."""
    for p in range(2, len(seq) // 2 + 1):
        if len(seq) % p == 0 and all(seq[i] == seq[i % p] for i in range(len(seq))):
            unit, reps = seq[:p], len(seq) // p
            break
    else:
        unit, reps = seq, 1
    folded: list[list] = []
    for label in unit:
        if folded and folded[-1][0] == label:
            folded[-1][1] += 1
        else:
            folded.append([label, 1])
    return folded, reps


def apply_names(lines: list[dict], bars: list[list[dict]], guessed: list[str], overrides: dict[str, str]) -> list[str]:
    """Guessed names with Lawrence's renames on top. `overrides` maps a line's
    start time ("12.3", seconds) to a name: from that line to the end of its
    guessed section, the lines take that name. Renaming a whole section is an
    override at its first line; "a new section starts here" is one mid-section."""
    names = list(guessed)
    if not overrides:
        return names
    times = [(float(t), nm) for t, nm in overrides.items()]
    for run in section_runs(lines, guessed):
        current = None
        for i in run["lines"]:
            start = bars[lines[i]["start"]][0]["start"]
            current = next((nm for t, nm in times if abs(t - start) <= 1.0), current)
            if current:
                names[i] = current
    return names


def song_sections(lines: list[dict], bars: list[list[dict]], names: list[str]) -> dict:
    """The song view's structure.

    runs:   [{name, start, end, block, reps, time}] in song order (a chorus sung
            twice in a row is one run with reps=2)
    blocks: [{title, name, lines: [[label, times in a row, [start time of that line in each run]]],
             count, times}] each distinct section
            written once; count = how many times it is played in total; times =
            start times of its runs (what a rename applies to)"""
    blocks: list[dict] = []
    runs = []
    for run in section_runs(lines, names):
        seq = [lines[i]["label"] for i in run["lines"]]
        unit, reps = _unit(seq)
        sig = (run["name"], tuple((_base(x), k) for x, k in unit))
        block = next((k for k, b in enumerate(blocks) if b["_sig"] == sig), None)
        time = round(bars[run["start"]][0]["start"], 1)
        if block is None:
            block = len(blocks)
            blocks.append({"_sig": sig, "name": run["name"], "lines": [[l, k, []] for l, k in unit],
                           "count": 0, "times": []})
        blocks[block]["count"] += reps
        blocks[block]["times"].append(time)
        # where each line of the section starts in this run (its first time through)
        pos = 0
        for entry, (_, k) in zip(blocks[block]["lines"], unit):
            entry[2].append(round(bars[lines[run["lines"][pos]]["start"]][0]["start"], 1))
            pos += k
        runs.append({"name": run["name"], "start": run["start"], "end": run["end"], "block": block,
                     "reps": reps, "time": time})
    seen: Counter[str] = Counter()
    for b in blocks:
        seen[b["name"]] += 1
        b["title"] = b["name"] + (f" ({seen[b['name']]})" if seen[b["name"]] > 1 else "")
        del b["_sig"]
    return {"runs": runs, "blocks": blocks}


def sections_for(sheet: dict, overrides: dict[str, str] | None = None) -> dict:
    """The song view for a sheet: its guessed names with `overrides` applied."""
    form = sheet.get("form") or {}
    lines, guessed = form.get("lines", []), form.get("guess")
    if not lines or not guessed:
        return {"runs": [], "blocks": []}
    return song_sections(lines, sheet["bars"], apply_names(lines, sheet["bars"], guessed, overrides or {}))
