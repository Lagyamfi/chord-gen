"""The musical pulse the player counts and clicks on, and the bar lines it implies.

Beat trackers' downbeats are mostly right but glitch: a stray downbeat makes a
1- or 7-beat bar, and half-time tracking makes 2-beat bars. A learner needs a
steady count, so we find the song's main bar length and number beats in a
regular cycle, changing phase only when two downbeats in a row agree."""

from __future__ import annotations

from collections import Counter

import numpy as np


def steady_beats(
    beats: list[float],
    phase_gain: float = 0.10,
    period_gain: float = 0.02,
    window: float = 0.2,
    max_drift: float = 0.12,
) -> np.ndarray:
    """A steady beat grid through detected beats: a phase-locked loop.

    Each step predicts the next beat one period ahead. A detection within
    `window` (fraction of a period) of the prediction pulls the beat a little
    toward it (`phase_gain`) and nudges the tempo (`period_gain`); anything else
    (a missed beat, a stray extra one) is ignored and the grid carries on. The
    period may drift at most `max_drift` from the song's median, so it follows a
    live band's slow push and pull without wobbling beat to beat. After a few
    beats with no matching detection the window widens to pick the band up again."""
    b = np.sort(np.asarray(beats, dtype=float))
    if len(b) < 3:
        return b
    ibi = np.diff(b)
    base = float(np.median(ibi))
    period = float(np.median(ibi[: min(8, len(ibi))]))
    period = base if abs(period - base) > max_drift * base else period
    grid = [float(b[0])]
    coasted = 0
    while True:
        pred = grid[-1] + period
        if pred > b[-1] + period / 2:
            break
        j = int(np.searchsorted(b, pred))
        near = min((x for x in b[max(0, j - 1) : j + 1]), key=lambda x: abs(x - pred))
        err = near - pred
        tol = window * period * (1 + min(coasted, 4) * 0.3)  # widen while lost
        if abs(err) <= tol:
            gain = 1.0 if coasted >= 4 else phase_gain  # re-lock firmly after a gap
            grid.append(pred + gain * err)
            period = float(np.clip(period + period_gain * err, base * (1 - max_drift), base * (1 + max_drift)))
            coasted = 0
        else:
            grid.append(pred)
            coasted += 1
    return np.array(grid)


def _downbeat_indexes(beats: np.ndarray, downbeats: list[float], tol: float) -> list[int]:
    if len(beats) == 0:
        return []
    return sorted({int(np.abs(beats - d).argmin()) for d in downbeats if np.abs(beats - d).min() <= tol})


def regular_pulse(beats: list[float], downbeats: list[float], tol: float = 0.12) -> list[dict]:
    """Each beat of the steady grid as {"t": seconds, "n": beat in bar (1-based), "of": beats per bar}.

    - meter = the most common gap between detected downbeats (ties go to 4)
    - a meter of 2 means the tracker counted half notes: beats are doubled
      (a midpoint added between each pair) and the meter becomes 4
    - numbering cycles 1..meter from a phase set by the first downbeat; the phase
      moves only when two consecutive downbeats agree on a new one, so a single
      stray downbeat cannot break the count
    With no usable downbeats every beat is 1 of 1."""
    b = steady_beats(beats)
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
