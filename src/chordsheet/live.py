"""Live mode: transcribe a growing recording chunk by chunk and keep a running sheet.

Chords appear about one chunk plus model time behind the music (~40 s with
30 s chunks): good for working out a song you cannot download, not karaoke."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from chordsheet.cache import DEFAULT_ROOT, Cache, sha256_text
from chordsheet.harmony.beats import LibrosaBeatTracker
from chordsheet.ingest.chunk import stitch_notes
from chordsheet.models import NoteEvent
from chordsheet.pipeline import build_sheet
from chordsheet.transcribe import get_transcriber

SILENCE_RMS = 1e-4


class LiveSession:
    def __init__(
        self,
        samplerate: int,
        transcriber: str = "muscriptor",
        model_size: str = "medium",
        chunk_s: float = 30.0,
        overlap_s: float = 2.0,
        cache_root: Path = DEFAULT_ROOT,
        title: str | None = None,
        capo: int | None = None,
        key: str | None = None,
    ):
        started = datetime.now()
        self.sr = samplerate
        self.chunk = int(chunk_s * samplerate)
        self.overlap = int(overlap_s * samplerate)
        self.source = title or f"Live {started:%Y-%m-%d %H:%M}"
        self.cache = Cache(sha256_text(f"live:{started.isoformat()}"), cache_root)
        self.label = f"muscriptor-{model_size}" if transcriber == "muscriptor" else transcriber
        kwargs = {"size": model_size} if transcriber == "muscriptor" else {}
        self.transcriber = get_transcriber(transcriber, **kwargs)  # kept: the model loads once
        self.capo = capo
        self.key = key
        self.audio = np.zeros(0, dtype=np.float32)
        self.done = 0  # samples transcribed so far
        self.chunks: list[tuple[float, list[NoteEvent]]] = []
        self.last_chunk_silent = False

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sr

    @property
    def pending_seconds(self) -> float:
        return (len(self.audio) - self.done) / self.sr

    def feed(self, block: np.ndarray) -> dict | None:
        """Add audio; returns an updated sheet whenever a full chunk is ready."""
        if block.ndim > 1:  # mix to mono
            block = block.mean(axis=1)
        if len(block):
            self.audio = np.concatenate([self.audio, block.astype(np.float32)])
        if len(self.audio) - self.done >= self.chunk:
            return self._process(self.done + self.chunk)
        return None

    def finish(self) -> dict | None:
        """Transcribe whatever is left (if at least a second)."""
        if len(self.audio) - self.done >= self.sr:
            return self._process(len(self.audio))
        return None

    def _process(self, end: int) -> dict:
        start = max(0, self.done - self.overlap)
        segment = self.audio[start:end]
        self.last_chunk_silent = float(np.sqrt(np.mean(segment**2))) < SILENCE_RMS
        chunk_path = self.cache.path("chunk.wav")
        sf.write(chunk_path, segment, self.sr)
        self.chunks.append((start / self.sr, self.transcriber.transcribe(chunk_path)))
        self.done = end

        notes = stitch_notes(self.chunks)
        sf.write(self.cache.path("audio.wav"), self.audio[:end], self.sr)
        self.cache.save_json(f"notes.{self.label}.json", [n.to_dict() for n in notes])
        grid = LibrosaBeatTracker().track(self.cache.path("audio.wav"))
        self.cache.save_json("beats.librosa.json", grid.to_dict())
        # refresh: the recording grew, so beat_this's bar grid must be redone too
        return build_sheet(self.cache, self.label, notes, grid, self.source, capo=self.capo, refresh=True, key=self.key)
