"""The musical pulse the player counts and clicks on, and the bar lines it implies.

Beat trackers' downbeats are mostly right but glitch: a stray downbeat makes a
1- or 7-beat bar, and half-time tracking makes 2-beat bars. A learner needs a
steady count, so we find the song's main bar length and number beats in a
regular cycle, changing phase only when two downbeats in a row agree."""

from __future__ import annotations

from collections import Counter

import numpy as np


def _downbeat_indexes(beats: np.ndarray, downbeats: list[float], tol: float) -> list[int]:
    if len(beats) == 0:
        return []
    return sorted({int(np.abs(beats - d).argmin()) for d in downbeats if np.abs(beats - d).min() <= tol})


def regular_pulse(beats: list[float], downbeats: list[float], tol: float = 0.07) -> list[dict]:
    """Each beat as {"t": seconds, "n": beat in bar (1-based), "of": beats per bar}.

    - meter = the most common gap between detected downbeats (ties go to 4)
    - a meter of 2 means the tracker counted half notes: beats are doubled
      (a midpoint added between each pair) and the meter becomes 4
    - numbering cycles 1..meter from a phase set by the first downbeat; the phase
      moves only when two consecutive downbeats agree on a new one, so a single
      stray downbeat cannot break the count
    With no usable downbeats every beat is 1 of 1."""
    b = np.asarray(beats, dtype=float)
    idx = _downbeat_indexes(b, downbeats, tol)
    if len(b) == 0:
        return []
    if len(idx) < 2:
        return [{"t": round(float(t), 3), "n": 1, "of": 1} for t in b]

    gaps = Counter(int(g) for g in np.diff(idx) if g > 0)
    meter = max(gaps, key=lambda g: (gaps[g], g == 4))
    if meter == 2:  # half-time tracking: count quarters, not halves
        mids = (b[:-1] + b[1:]) / 2
        b = np.sort(np.concatenate([b, mids]))
        idx = [2 * i for i in idx]
        meter = 4

    phase = idx[0]
    starts = {phase: phase}  # beat index -> phase in force from there
    for prev, cur, nxt in zip(idx, idx[1:], [*idx[2:], None]):
        if (cur - phase) % meter and nxt is not None and (nxt - cur) % meter == 0:
            phase = cur
            starts[cur] = phase
    out = []
    phase = idx[0]
    for i, t in enumerate(b):
        phase = starts.get(i, phase)
        out.append({"t": round(float(t), 3), "n": (i - phase) % meter + 1, "of": meter})
    return out


def bar_starts(pulse: list[dict]) -> list[float]:
    """Downbeat times implied by a regular pulse."""
    return [p["t"] for p in pulse if p["n"] == 1]
