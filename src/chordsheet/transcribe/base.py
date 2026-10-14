from __future__ import annotations

from pathlib import Path
from typing import Protocol

from chordsheet.models import NoteEvent


class Transcriber(Protocol):
    name: str

    def transcribe(self, wav_path: Path) -> list[NoteEvent]: ...


def get_transcriber(name: str, **kwargs) -> Transcriber:
    if name == "chroma":
        from chordsheet.transcribe.chroma_fallback import ChromaTranscriber

        return ChromaTranscriber(**kwargs)
    if name == "muscriptor":
        from chordsheet.transcribe.muscriptor import MuScriptorTranscriber

        return MuScriptorTranscriber(**kwargs)
    raise ValueError(f"unknown transcriber: {name}")
