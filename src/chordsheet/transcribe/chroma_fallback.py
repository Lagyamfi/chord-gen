"""Baseline transcriber: spectral chroma turned into pseudo note events.

Worse than a real transcription model by design; it keeps the pipeline testable
without model weights and is the baseline in the evaluation.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from chordsheet.models import NoteEvent

CHROMA_INSTRUMENT = "chroma"  # pitch-class-only pseudo notes; never used as bass
BASS_INSTRUMENT = "bass"


class ChromaTranscriber:
    name = "chroma"

    def __init__(
        self,
        sr: int = 22050,
        hop: int = 2048,
        pc_threshold: float = 0.15,
        silence_db: float = -40.0,
        bass_threshold: float = 0.25,
    ):
        self.sr = sr
        self.hop = hop
        self.pc_threshold = pc_threshold
        self.silence_db = silence_db
        self.bass_threshold = bass_threshold

    def transcribe(self, wav_path: Path) -> list[NoteEvent]:
        y, sr = librosa.load(str(wav_path), sr=self.sr, mono=True)
        harm = librosa.effects.harmonic(y)
        # Upper harmony from C3 up, so a loud bass line does not drown out the third.
        chroma = librosa.feature.chroma_cqt(
            y=harm, sr=sr, hop_length=self.hop, fmin=librosa.note_to_hz("C3"), n_octaves=4
        )  # (12, T), max-normalised per frame
        rms = librosa.feature.rms(y=harm, frame_length=self.hop * 2, hop_length=self.hop)[0]
        loud = librosa.amplitude_to_db(rms, ref=np.max) > self.silence_db

        # Bass: strongest CQT bin between C1 and B2.
        bass_cqt = np.abs(
            librosa.cqt(harm, sr=sr, hop_length=self.hop, fmin=librosa.note_to_hz("C1"), n_bins=24)
        )
        bass_peak = bass_cqt.max(axis=0)
        bass_ok = bass_peak > self.bass_threshold * (np.percentile(bass_peak, 95) + 1e-9)

        n = min(chroma.shape[1], len(loud), bass_cqt.shape[1])
        dt = self.hop / sr
        notes: list[NoteEvent] = []
        for t in range(n):
            if not loud[t]:
                continue
            on, off = t * dt, (t + 1) * dt
            for pc in np.flatnonzero(chroma[:, t] >= self.pc_threshold):
                notes.append(NoteEvent(60 + int(pc), on, off, float(chroma[pc, t]), CHROMA_INSTRUMENT))
            if bass_ok[t]:
                pitch = 24 + int(bass_cqt[:, t].argmax())
                notes.append(NoteEvent(pitch, on, off, 1.0, BASS_INSTRUMENT))
        return notes
