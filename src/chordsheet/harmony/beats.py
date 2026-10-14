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


class BeatThisTracker:
    """beat_this (Foscarin et al., ISMIR 2024), the tracker MuScriptor uses
    internally. Unlike librosa it detects downbeats, so bar lines come from the
    audio. We keep the raw tracked beats (no constant-tempo fit), so live tempo
    drift is followed. Fast enough on CPU."""

    def __init__(self, device: str = "cpu", checkpoint: str = "final0"):
        self.device = device
        self.checkpoint = checkpoint

    def track(self, wav_path: Path) -> BeatGrid:
        import soundfile as sf
        from beat_this.inference import Audio2Beats

        y, sr = sf.read(str(wav_path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        beats, downbeats = Audio2Beats(checkpoint_path=self.checkpoint, device=self.device, dbn=False)(y, sr)
        beats = np.asarray(beats, dtype=float)
        tempo = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 1 else 0.0
        # snap each downbeat to its nearest tracked beat so bars split cleanly
        snapped = sorted({float(beats[np.abs(beats - d).argmin()]) for d in downbeats}) if len(beats) else []
        return grid_from_beats(beats, tempo, len(y) / sr, downbeats=snapped)


BEAT_TRACKERS = {"librosa": LibrosaBeatTracker, "beat-this": BeatThisTracker}


def get_beat_tracker(name: str) -> BeatTracker:
    if name not in BEAT_TRACKERS:
        raise ValueError(f"unknown beat tracker: {name} (choose from {', '.join(BEAT_TRACKERS)})")
    return BEAT_TRACKERS[name]()


def grid_from_beats(
    beats, tempo: float, duration: float, beats_per_bar: int = 4, downbeats: list[float] | None = None
) -> BeatGrid:
    """Beats -> BeatGrid. Without detected `downbeats`, bars are assumed every
    `beats_per_bar` beats from the first beat."""
    beats = [float(b) for b in beats]
    if len(beats) < 2:  # no usable beat: one window over the whole track
        return BeatGrid([0.0], [0.0], tempo, duration)
    ibi = float(np.median(np.diff(beats)))
    end = min(beats[-1] + ibi, duration) if duration > beats[-1] else beats[-1] + ibi
    if downbeats:
        return BeatGrid(beats, list(downbeats), tempo, end, has_downbeats=True)
    return BeatGrid(beats, beats[::beats_per_bar], tempo, end)
