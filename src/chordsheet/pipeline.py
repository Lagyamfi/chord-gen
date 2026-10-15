"""Wires the layers together with the per-song cache."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from chordsheet.cache import DEFAULT_ROOT, Cache
from chordsheet.guitar.bars import best_bar_phase, downbeats_from_phase
from chordsheet.guitar.form import song_form
from chordsheet.guitar.sections import guess_sections, line_energy, sections_for
from chordsheet.guitar.render import group_bars
from chordsheet.guitar.rhythm import bar_starts, regular_pulse
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
    progress: Callable[[str], None] | None = None,
) -> tuple[Cache, str, list[NoteEvent], BeatGrid]:
    """Layers 1-2 plus the beat grid, cached. Returns (cache, transcriber label, notes, grid).
    `progress` is told each stage as it starts: download, transcribe, beats."""
    step = progress or (lambda stage: None)
    step("download")
    cache, wav = ingest(source, cache_root=cache_root)
    # cache label: model sizes produce different notes, so each gets its own file
    label = f"muscriptor-{model_size}" if transcriber == "muscriptor" else transcriber

    def _transcribe():
        kwargs = {}
        if transcriber == "muscriptor":
            kwargs = {"size": model_size, "midi_out": cache.path(f"transcription.{label}.mid")}
        return [n.to_dict() for n in get_transcriber(transcriber, **kwargs).transcribe(wav)]

    step("transcribe")
    notes = [NoteEvent.from_dict(d) for d in cache.get_or_compute(f"notes.{label}.json", _transcribe, refresh)]
    step("beats")
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
    key: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """`progress` is told each stage as it starts: download, transcribe, beats, chords.
    `beats`: "librosa", "beat-this", or "hybrid" (default): chords analysed on
    librosa's beats, bars and tempo from beat_this. librosa often tracks at double
    tempo, and its shorter windows place chord changes ~1 point more accurately on
    the real set; beat_this finds the true tempo and real downbeats."""
    harmony_beats = "librosa" if beats == "hybrid" else beats
    cache, label, notes, grid = load_layers(source, transcriber, model_size, cache_root, refresh, harmony_beats, progress)
    return build_sheet(cache, label, notes, grid, source, self_transition, rich, beats, capo, refresh, key, progress)


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
    key: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Layers 3-4 from cached notes and beats: harmony, bars, capo and shapes.
    Writes chords.json, sheet.json and sheet.<label>.json into the cache."""
    harmony = analyse_harmony(notes, grid, self_transition=self_transition, rich=rich, key=key).to_dict()
    cache.save_json("chords.json", harmony)

    tempo = grid.tempo
    if beats == "hybrid":
        bar_grid = BeatGrid.from_dict(
            cache.get_or_compute(
                "beats.beat-this.json", lambda: get_beat_tracker("beat-this").track(cache.path("audio.wav")).to_dict(), refresh
            )
        )
        # count, click and bar lines all come from beat_this's beats made regular:
        # the musical pulse (librosa's harmony beats often run at double tempo)
        pulse = regular_pulse(bar_grid.beats, bar_grid.downbeats)
        downbeats = snap_to_beats(bar_starts(pulse), grid.beats)
        tempo = bar_grid.tempo * (2 if len(pulse) > 1.5 * len(bar_grid.beats) else 1)
    elif grid.has_downbeats:
        pulse = regular_pulse(grid.beats, grid.downbeats)
        downbeats = bar_starts(pulse)
    else:  # no downbeat model (librosa): align bars so chord changes fall on beat 1
        phase = best_bar_phase([b["chord"] for b in harmony["beats"]])
        downbeats = downbeats_from_phase(grid.beats, phase)
        pulse = regular_pulse(grid.beats, downbeats)

    if progress:
        progress("chords")
    bars = group_bars(harmony["beats"], downbeats)
    guitar = guitarify(bars, capo)
    form = song_form(bars)
    if form["lines"]:
        energy = line_energy(bars, form["lines"], cache.path("audio.wav"), notes)
        form["guess"] = guess_sections(form["lines"], energy)

    sheet = {
        "source": source,
        "cache_key": cache.key,
        "transcriber": label,
        "key": harmony["key"],
        "key_source": harmony["key_source"],
        "key_alternatives": harmony["key_alternatives"],
        "tempo": tempo,
        "beat_tracker": beats,
        **guitar,
        "pulse": pulse,
        "form": form,
        "bars": bars,
    }
    names = cache.load_json("names.json") if cache.has("names.json") else {}
    sheet["sections"] = sections_for(sheet, names)  # the server re-applies renames on every request
    cache.save_json("sheet.json", sheet)  # latest run
    cache.save_json(f"sheet.{label}.json", sheet)  # per system, for the player
    return sheet
