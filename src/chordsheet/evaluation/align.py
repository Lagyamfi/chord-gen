"""Forced alignment of a known chord sequence (from a chart) to the audio's beats.

The chart decides WHICH chords occur and in what order; the audio only decides
WHEN each change happens. A left-to-right HMM over the sequence, with optional
no-chord at either end, decoded with Viterbi. All 12 transpositions are tried,
because charts are often in a different key from the recording.

Section loops (a section jumping back to its own start, for live repeats) are
off by default: when the recording departs from the chart, loops let the
aligner repeat whichever section happens to contain the right chords and wreck
the structure. Better to keep the chart's order and flag low-fit stretches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chordsheet.evaluation.chart import ChartChord, SeqItem

NEG = -1e18


def chord_template(chord: ChartChord) -> np.ndarray:
    import mir_eval

    if chord.is_none:
        return np.zeros(12)
    _, bitmap, _ = mir_eval.chord.encode(chord.label)
    v = np.roll(bitmap.astype(float), chord.root)
    return v / np.linalg.norm(v)


def emission(
    chords: list[ChartChord],
    profiles: np.ndarray,
    bass_pc: np.ndarray,
    bass_bonus: float = 0.15,
    no_chord_score: float = 0.2,
) -> np.ndarray:
    """(n_beats, len(chords) + 1) scores; the last column is no-chord."""
    templates = np.array([chord_template(c) for c in chords])
    scores = profiles @ templates.T
    silent = ~profiles.any(axis=1)
    no_chord = np.where(silent, 1.0, no_chord_score)
    for j, c in enumerate(chords):
        if c.is_none:  # "N.C." in the chart
            scores[:, j] = no_chord
        else:
            scores[:, j] += np.where(bass_pc == (c.root + c.bass) % 12, bass_bonus, 0.0)
    return np.column_stack([scores, no_chord])


def section_loops(section_index: list[int]) -> list[tuple[int, int]]:
    """(last state, first state) of each section, in sequence positions."""
    loops = []
    for idx in sorted(set(section_index)):
        pos = [i for i, s in enumerate(section_index) if s == idx]
        if len(pos) > 1:
            loops.append((pos[-1], pos[0]))
    return loops


def left_to_right_viterbi(
    emit: np.ndarray,
    loops: list[tuple[int, int]] = (),
    anchors: list[tuple[int, int]] = (),
    scale: float = 20.0,
    loop_penalty: float = 3.0,
) -> tuple[np.ndarray, float]:
    """States: 0 = lead-in N, 1..K = sequence chords, K+1 = tail N. Each beat
    stays, advances one state, or (from a section's last chord) jumps back to the
    section's first chord at `loop_penalty`. Each (position, beat) anchor forces
    the path to enter that sequence position exactly at that beat.
    Returns (state per beat, score)."""
    n, k1 = emit.shape
    k = k1 - 1
    e = np.empty((n, k + 2))
    e[:, 0] = e[:, k + 1] = emit[:, k]
    e[:, 1 : k + 1] = emit[:, :k]
    e *= scale
    for pos, beat in anchors:
        state = pos + 1
        e[:beat, state:] = NEG
        e[beat, :] = NEG
        e[beat, state] = emit[beat, pos] * scale
        e[beat + 1 :, :state] = NEG
    loop_src = np.array([a + 1 for a, _ in loops], dtype=int)  # +1: state offset for lead-in N
    loop_dst = np.array([b + 1 for _, b in loops], dtype=int)

    delta = np.full(k + 2, NEG)
    delta[0], delta[1] = e[0, 0], e[0, 1]
    back = np.zeros((n, k + 2), dtype=np.int32)
    states = np.arange(k + 2)
    for t in range(1, n):
        best = delta.copy()
        src = states.copy()
        advance = np.concatenate([[NEG], delta[:-1]])
        better = advance > best
        best[better], src[better] = advance[better], states[better] - 1
        if len(loops):
            jump = delta[loop_src] - loop_penalty
            better = jump > best[loop_dst]
            best[loop_dst[better]], src[loop_dst[better]] = jump[better], loop_src[better]
        back[t] = src
        delta = best + e[t]

    end = k + 1 if delta[k + 1] >= delta[k] else k
    path = np.empty(n, dtype=np.int32)
    path[-1] = end
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(delta[end])


@dataclass
class Alignment:
    transpose: int  # semitones applied to the chart to match the audio
    labels: list[str]  # per beat, mir_eval labels ("N" outside the sequence)
    sections: list[str]  # per beat, chart section name ("" outside)
    fit: np.ndarray  # per beat, raw emission score of the assigned chord
    loops: list[tuple[str, float]]  # (section name, time) where a section repeated
    candidates: list[tuple[int, float]]  # (transpose, mean fit) for all 12, best first


def align(
    sequence: list[SeqItem],
    profiles: np.ndarray,
    bass_pc: np.ndarray,
    transpose: int | None = None,
    allow_loops: bool = False,
    times: list[float] | None = None,
    anchors: list[tuple[int, int]] = (),
) -> Alignment:
    """`anchors`: (sequence position, beat index) pairs the path must hit."""
    if len(sequence) > len(profiles):
        raise ValueError(f"chart has {len(sequence)} chord changes but the song only {len(profiles)} beats")
    loops = section_loops([it.section_index for it in sequence]) if allow_loops else []
    shifts = range(12) if transpose is None else [transpose]
    results = []
    for shift in shifts:
        chords = [it.chord.transpose(shift) for it in sequence]
        emit = emission(chords, profiles, bass_pc)
        path, _ = left_to_right_viterbi(emit, loops, anchors)
        k = len(chords)
        cols = np.where((path == 0) | (path == k + 1), k, path - 1)
        fit = emit[np.arange(len(path)), cols]
        results.append((float(fit.mean()), shift, path, chords, fit))
    results.sort(key=lambda r: -r[0])
    _, shift, path, chords, fit = results[0]
    k = len(chords)
    labels = ["N" if s in (0, k + 1) else chords[s - 1].label for s in path]
    sections = ["" if s in (0, k + 1) else sequence[s - 1].section for s in path]
    jumps = [
        (sequence[path[t] - 1].section, times[t] if times else float(t))
        for t in range(1, len(path))
        if path[t] < path[t - 1]
    ]
    return Alignment(shift, labels, sections, fit, jumps, [(r[1], round(r[0], 3)) for r in results])
