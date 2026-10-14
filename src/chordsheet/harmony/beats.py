from __future__ import annotations

from pathlib import Path
from typing import Protocol

import librosa
import numpy as np

from chordsheet.models import BeatGrid


class BeatTracker(Protocol):
    def track(self, wav_path: Path) -> BeatGrid: ...


class LibrosaBeatTracker:
    """librosa beat tracking. It has no downbeat model, so bars are assumed to
    start on the first beat and last `beats_per_bar` beats. madmom's
    DBNDownBeatTrackingProcessor can replace this behind the same protocol."""

    def __init__(self, sr: int = 22050, beats_per_bar: int = 4):
        self.sr = sr
        self.beats_per_bar = beats_per_bar

    def track(self, wav_path: Path) -> BeatGrid:
        y, sr = librosa.load(str(wav_path), sr=self.sr, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        return grid_from_beats(beats, float(np.atleast_1d(tempo)[0]), len(y) / sr, self.beats_per_bar)


def grid_from_beats(beats, tempo: float, duration: float, beats_per_bar: int = 4) -> BeatGrid:
    beats = [float(b) for b in beats]
    if len(beats) < 2:  # no usable beat: one window over the whole track
        return BeatGrid([0.0], [0.0], tempo, duration)
    ibi = float(np.median(np.diff(beats)))
    end = min(beats[-1] + ibi, duration) if duration > beats[-1] else beats[-1] + ibi
    return BeatGrid(beats, beats[::beats_per_bar], tempo, end)
