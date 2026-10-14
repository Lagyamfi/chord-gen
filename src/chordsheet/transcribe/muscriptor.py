"""MuScriptor (Kyutai / Mirelo AI / IRCAM) multi-instrument transcription.

Install with `uv sync --extra muscriptor`. The HF repo is gated: accept the
conditions at https://huggingface.co/MuScriptor/muscriptor-large and run
`uv run hf auth login` once. Runs on MPS automatically on Apple Silicon.
Weights are CC-BY-NC: personal use only.
"""

from __future__ import annotations

import io
from pathlib import Path

import pretty_midi

from chordsheet.models import NoteEvent


class MuScriptorTranscriber:
    name = "muscriptor"

    def __init__(self, size: str = "large", midi_out: Path | None = None):
        self.size = size
        self.midi_out = midi_out
        self._model = None

    def _load(self):
        if self._model is None:
            from muscriptor import TranscriptionModel

            self._model = TranscriptionModel.load_model(self.size)
        return self._model

    def transcribe(self, wav_path: Path) -> list[NoteEvent]:
        model = self._load()
        # no beat grid: keep note times in seconds, unquantised
        midi_bytes = model.events_to_midi_bytes(model.transcribe(str(wav_path)))
        if self.midi_out is not None:
            Path(self.midi_out).write_bytes(midi_bytes)
        return midi_to_notes(pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes)))


def instrument_label(inst: pretty_midi.Instrument) -> str:
    if inst.is_drum:
        return "drums"
    if inst.name:
        return inst.name.strip().lower()
    return pretty_midi.program_to_instrument_class(inst.program).lower()


def midi_to_notes(pm: pretty_midi.PrettyMIDI) -> list[NoteEvent]:
    notes = [
        NoteEvent(
            n.pitch,
            float(n.start),
            float(n.end),
            float(n.velocity),
            instrument_label(inst),
            inst.is_drum,
        )
        for inst in pm.instruments
        for n in inst.notes
    ]
    return sorted(notes, key=lambda n: (n.onset, n.pitch))
