"""Fixed-length chunking with overlap, plus note de-duplication across boundaries.

MuScriptor windows audio internally, so file mode does not need this; live mode
(week 4) and any model with a bounded input window does.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from chordsheet.models import NoteEvent


def chunk_audio(
    y: np.ndarray, sr: int, length_s: float = 30.0, overlap_s: float = 2.0
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (start_time_s, samples) windows."""
    step = int((length_s - overlap_s) * sr)
    size = int(length_s * sr)
    for start in range(0, max(len(y) - int(overlap_s * sr), 1), step):
        yield start / sr, y[start : start + size]


def stitch_notes(chunks: list[tuple[float, list[NoteEvent]]], tol: float = 0.05) -> list[NoteEvent]:
    """Merge per-chunk notes (chunk-relative times) into one list, dropping
    duplicates from the overlap region (same pitch + instrument, onset within `tol`)."""
    merged: list[NoteEvent] = []
    for start, notes in chunks:
        for n in notes:
            shifted = NoteEvent(n.pitch, n.onset + start, n.offset + start, n.velocity, n.instrument, n.is_drum)
            dup = any(
                m.pitch == shifted.pitch
                and m.instrument == shifted.instrument
                and abs(m.onset - shifted.onset) < tol
                for m in merged[-64:]
            )
            if not dup:
                merged.append(shifted)
    return sorted(merged, key=lambda n: (n.onset, n.pitch))
