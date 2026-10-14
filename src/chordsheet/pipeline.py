"""Wires the layers together with the per-song cache."""

from __future__ import annotations

from pathlib import Path

from chordsheet.cache import DEFAULT_ROOT
from chordsheet.guitar.bars import best_bar_phase, downbeats_from_phase
from chordsheet.guitar.render import group_bars
from chordsheet.harmony.analyse import analyse_harmony
from chordsheet.harmony.beats import LibrosaBeatTracker
from chordsheet.ingest import ingest
from chordsheet.models import BeatGrid, NoteEvent
from chordsheet.transcribe import get_transcriber


def analyse(
    source: str,
    transcriber: str = "chroma",
    self_transition: float = 0.85,
    sevenths: bool = True,
    cache_root: Path = DEFAULT_ROOT,
    refresh: bool = False,
) -> dict:
    cache, wav = ingest(source, cache_root=cache_root)

    def _transcribe():
        kwargs = {"midi_out": cache.path("transcription.mid")} if transcriber == "muscriptor" else {}
        return [n.to_dict() for n in get_transcriber(transcriber, **kwargs).transcribe(wav)]

    notes = [NoteEvent.from_dict(d) for d in cache.get_or_compute(f"notes.{transcriber}.json", _transcribe, refresh)]
    grid = BeatGrid.from_dict(cache.get_or_compute("beats.json", lambda: LibrosaBeatTracker().track(wav).to_dict(), refresh))

    harmony = analyse_harmony(notes, grid, self_transition=self_transition, sevenths=sevenths).to_dict()
    cache.save_json("chords.json", harmony)

    # librosa has no downbeats: align bars so chord changes fall on beat 1
    phase = best_bar_phase([b["chord"] for b in harmony["beats"]])
    downbeats = downbeats_from_phase(grid.beats, phase)

    sheet = {
        "source": source,
        "cache_key": cache.key,
        "transcriber": transcriber,
        "key": harmony["key"],
        "key_alternatives": harmony["key_alternatives"],
        "tempo": grid.tempo,
        "bars": group_bars(harmony["beats"], downbeats),
    }
    cache.save_json("sheet.json", sheet)
    return sheet
