"""Wires the layers together with the per-song cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chordsheet.cache import DEFAULT_ROOT, Cache
from chordsheet.guitar.bars import best_bar_phase, downbeats_from_phase
from chordsheet.guitar.render import group_bars
from chordsheet.guitar.sheet import guitarify
from chordsheet.harmony.analyse import analyse_harmony
from chordsheet.harmony.beats import get_beat_tracker
from chordsheet.ingest import ingest
from chordsheet.models import BeatGrid, NoteEvent
from chordsheet.transcribe import get_transcriber


def snap_to_beats(times: list[float], beats: list[float]) -> list[float]:
    """Move each time onto its nearest beat (deduplicated), so bars split cleanly."""
    if not beats:
        return []
    b = np.array(beats)
    return sorted({float(b[np.abs(b - t).argmin()]) for t in times})


def load_layers(
    source: str,
    transcriber: str = "chroma",
    model_size: str = "medium",
    cache_root: Path = DEFAULT_ROOT,
    refresh: bool = False,
    beats: str = "librosa",
) -> tuple[Cache, str, list[NoteEvent], BeatGrid]:
    """Layers 1-2 plus the beat grid, cached. Returns (cache, transcriber label, notes, grid)."""
    cache, wav = ingest(source, cache_root=cache_root)
    # cache label: model sizes produce different notes, so each gets its own file
    label = f"muscriptor-{model_size}" if transcriber == "muscriptor" else transcriber

    def _transcribe():
        kwargs = {}
        if transcriber == "muscriptor":
            kwargs = {"size": model_size, "midi_out": cache.path(f"transcription.{label}.mid")}
        return [n.to_dict() for n in get_transcriber(transcriber, **kwargs).transcribe(wav)]

    notes = [NoteEvent.from_dict(d) for d in cache.get_or_compute(f"notes.{label}.json", _transcribe, refresh)]
    grid = BeatGrid.from_dict(
        cache.get_or_compute(f"beats.{beats}.json", lambda: get_beat_tracker(beats).track(wav).to_dict(), refresh)
    )
    return cache, label, notes, grid


def analyse(
    source: str,
    transcriber: str = "chroma",
    self_transition: float = 0.85,
    rich: bool = False,
    cache_root: Path = DEFAULT_ROOT,
    refresh: bool = False,
    model_size: str = "medium",
    beats: str = "hybrid",
    capo: int | None = None,
) -> dict:
    """`beats`: "librosa", "beat-this", or "hybrid" (default): chords analysed on
    librosa's beats, bars and tempo from beat_this. librosa often tracks at double
    tempo, and its shorter windows place chord changes ~1 point more accurately on
    the real set; beat_this finds the true tempo and real downbeats."""
    harmony_beats = "librosa" if beats == "hybrid" else beats
    cache, label, notes, grid = load_layers(source, transcriber, model_size, cache_root, refresh, harmony_beats)
    return build_sheet(cache, label, notes, grid, source, self_transition, rich, beats, capo, refresh)


def build_sheet(
    cache: Cache,
    label: str,
    notes: list[NoteEvent],
    grid: BeatGrid,
    source: str,
    self_transition: float = 0.85,
    rich: bool = False,
    beats: str = "hybrid",
    capo: int | None = None,
    refresh: bool = False,
) -> dict:
    """Layers 3-4 from cached notes and beats: harmony, bars, capo and shapes.
    Writes chords.json, sheet.json and sheet.<label>.json into the cache."""
    harmony = analyse_harmony(notes, grid, self_transition=self_transition, rich=rich).to_dict()
    cache.save_json("chords.json", harmony)

    tempo = grid.tempo
    if beats == "hybrid":
        bar_grid = BeatGrid.from_dict(
            cache.get_or_compute(
                "beats.beat-this.json", lambda: get_beat_tracker("beat-this").track(cache.path("audio.wav")).to_dict(), refresh
            )
        )
        downbeats = snap_to_beats(bar_grid.downbeats, grid.beats)
        tempo = bar_grid.tempo
    elif grid.has_downbeats:
        downbeats = grid.downbeats
    else:  # no downbeat model (librosa): align bars so chord changes fall on beat 1
        phase = best_bar_phase([b["chord"] for b in harmony["beats"]])
        downbeats = downbeats_from_phase(grid.beats, phase)

    bars = group_bars(harmony["beats"], downbeats)
    guitar = guitarify(bars, capo)

    sheet = {
        "source": source,
        "cache_key": cache.key,
        "transcriber": label,
        "key": harmony["key"],
        "key_alternatives": harmony["key_alternatives"],
        "tempo": tempo,
        "beat_tracker": beats,
        **guitar,
        "bars": bars,
    }
    cache.save_json("sheet.json", sheet)  # latest run
    cache.save_json(f"sheet.{label}.json", sheet)  # per system, for the player
    return sheet
