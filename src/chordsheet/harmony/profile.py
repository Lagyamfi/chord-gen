"""Per-beat pitch-class profiles built from note events."""

from __future__ import annotations

import numpy as np

from chordsheet.models import NoteEvent

# Instrument weights for harmony, matched by substring of the instrument label.
# First match wins; tune per genre.
INSTRUMENT_WEIGHTS: list[tuple[str, float]] = [
    ("drum", 0.0),
    ("percussion", 0.0),
    ("voice", 0.5),
    ("vocal", 0.5),
    ("choir", 0.5),
    ("bass", 1.0),
    ("guitar", 1.0),
    ("piano", 1.0),
    ("organ", 1.0),
    ("string", 1.0),
    ("pad", 1.0),
    ("synth", 0.8),
    ("chroma", 1.0),
]
DEFAULT_WEIGHT = 0.7

# Pseudo-note sources whose pitch octave is meaningless; never used as bass.
NO_BASS_INSTRUMENTS = {"chroma"}


def instrument_weight(label: str, table=INSTRUMENT_WEIGHTS, default: float = DEFAULT_WEIGHT) -> float:
    label = label.lower()
    for key, w in table:
        if key in label:
            return w
    return default


def beat_profiles(
    notes: list[NoteEvent],
    windows: list[tuple[float, float]],
    weights=INSTRUMENT_WEIGHTS,
    bass_min_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (profiles (n, 12) unit-norm or zero, bass_pc (n,) with -1 for none).

    Each note overlapping a window adds overlap_s * velocity * instrument_weight to
    bin pitch % 12. The bass is the lowest note covering at least
    `bass_min_fraction` of the window."""
    n = len(windows)
    profiles = np.zeros((n, 12))
    bass_pitch = np.full(n, 10_000)
    starts = np.array([w[0] for w in windows])

    for note in notes:
        if note.is_drum:
            continue
        w = instrument_weight(note.instrument, weights)
        if w <= 0:
            continue
        # first window that could overlap this note
        i = max(int(np.searchsorted(starts, note.onset, side="right")) - 1, 0)
        while i < n and windows[i][0] < note.offset:
            b0, b1 = windows[i]
            overlap = min(note.offset, b1) - max(note.onset, b0)
            if overlap > 0:
                profiles[i, note.pitch % 12] += overlap * note.velocity * w
                if (
                    note.instrument not in NO_BASS_INSTRUMENTS
                    and overlap >= bass_min_fraction * (b1 - b0)
                    and note.pitch < bass_pitch[i]
                ):
                    bass_pitch[i] = note.pitch
            i += 1

    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles = np.divide(profiles, norms, out=np.zeros_like(profiles), where=norms > 0)
    bass_pc = np.where(bass_pitch < 10_000, bass_pitch % 12, -1)
    return profiles, bass_pc
