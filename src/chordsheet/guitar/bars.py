"""Bar alignment for beat trackers without a downbeat model."""

from __future__ import annotations


def best_bar_phase(chords: list[str], beats_per_bar: int = 4) -> int:
    """Beat offset (0..beats_per_bar-1) at which the most chord changes land on
    beat 1. Chord changes overwhelmingly happen on downbeats, so this recovers the
    bar phase from the harmony itself."""
    counts = [0] * beats_per_bar
    for i in range(1, len(chords)):
        if chords[i] != chords[i - 1]:
            counts[i % beats_per_bar] += 1
    return max(range(beats_per_bar), key=lambda p: (counts[p], -p))


def downbeats_from_phase(beats: list[float], phase: int, beats_per_bar: int = 4) -> list[float]:
    return beats[phase::beats_per_bar] if phase else beats[::beats_per_bar]
