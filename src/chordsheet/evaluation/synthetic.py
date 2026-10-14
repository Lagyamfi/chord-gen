"""Synthetic evaluation set: known progressions rendered through a soundfont.

Ground truth is exact, so a miss here is a bug in layer 3 (or the transcriber),
never ambiguity in the music."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

from chordsheet.harmony.templates import NOTE_NAMES, QUALITIES, Chord

# name -> chord symbols, one bar each
PROGRESSIONS: dict[str, list[str]] = {
    "g_d_em_c": ["G", "D", "Em", "C"],
    "c_g_am_f": ["C", "G", "Am", "F"],
    "am_f_c_g": ["Am", "F", "C", "G"],
    "d_a_bm_g": ["D", "A", "Bm", "G"],
    "e_b_csm_a": ["E", "B", "C#m", "A"],
    "f_c_dm_bb": ["F", "C", "Dm", "Bb"],
    "dm_bb_f_c": ["Dm", "Bb", "F", "C"],
    "a_e_fsm_d": ["A", "E", "F#m", "D"],
    "em_c_g_d": ["Em", "C", "G", "D"],
    "bb_f_gm_eb": ["Bb", "F", "Gm", "Eb"],
}

_ALIASES = {"C#": 1, "Db": 1, "D#": 3, "Gb": 6, "G#": 8, "A#": 10}


def parse_symbol(symbol: str) -> Chord:
    """'C#m' -> Chord(1, 'min'). Handles the triad and seventh suffixes in QUALITIES."""
    root_len = 2 if len(symbol) > 1 and symbol[1] in "#b" else 1
    root_name, suffix = symbol[:root_len], symbol[root_len:]
    root = _ALIASES.get(root_name, NOTE_NAMES.index(root_name) if root_name in NOTE_NAMES else -1)
    if root < 0:
        raise ValueError(symbol)
    for q, (_, sfx, _) in QUALITIES.items():
        if sfx == suffix:
            return Chord(root, q)
    raise ValueError(symbol)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    chord: Chord


def build_midi(symbols: list[str], bpm: float = 100.0, repeats: int = 2, drums: bool = True):
    """Piano block chords on every beat, root bass on beats 1 and 3, simple rock beat.
    Starts with one bar of count-in drums so the beat tracker locks on.
    Returns (PrettyMIDI, ground-truth segments)."""
    beat = 60.0 / bpm
    bar = 4 * beat
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name="piano")
    bass = pretty_midi.Instrument(program=33, name="bass")
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    segments: list[Segment] = [Segment(0.0, bar, Chord(-1, "N"))]

    t = bar  # count-in bar
    for _ in range(repeats):
        for sym in symbols:
            chord = parse_symbol(sym)
            # root-position voicing, root between F#3 and F4
            base = 60 + chord.root - (12 if chord.root >= 6 else 0)
            pitches = [base + i for i in chord.intervals]
            for b in range(4):
                on = t + b * beat
                for p in pitches:
                    piano.notes.append(pretty_midi.Note(80, p, on, on + beat * 0.95))
            for b in (0, 2):
                on = t + b * beat
                bass.notes.append(pretty_midi.Note(95, 36 + chord.root, on, on + 2 * beat * 0.95))
            segments.append(Segment(t, t + bar, chord))
            t += bar

    if drums:
        for b in range(int(round(t / beat))):
            on = b * beat
            kit.notes.append(pretty_midi.Note(70, 42, on, on + 0.1))  # hi-hat
            kit.notes.append(pretty_midi.Note(90, 36 if b % 2 == 0 else 38, on, on + 0.1))  # kick / snare
        pm.instruments.append(kit)
    pm.instruments.extend([piano, bass])
    segments.append(Segment(t, t + 10.0, Chord(-1, "N")))  # release tail
    return pm, segments


def find_soundfont() -> Path:
    env = os.environ.get("CHORDSHEET_SOUNDFONT")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[3]
    for pattern in ("assets/soundfonts/*.sf2", "assets/soundfonts/*.sf3"):
        found = sorted(root.glob(pattern))
        if found:
            return found[0]
    brew = sorted(glob.glob("/opt/homebrew/Cellar/fluid-synth/*/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2"))
    if brew:
        return Path(brew[-1])
    raise FileNotFoundError("no soundfont: set CHORDSHEET_SOUNDFONT or put one in assets/soundfonts/")


def render_wav(midi_path: Path, wav_path: Path, soundfont: Path | None = None, sr: int = 44100) -> None:
    if shutil.which("fluidsynth") is None:
        raise RuntimeError("fluidsynth not installed (brew install fluid-synth)")
    sf = soundfont or find_soundfont()
    subprocess.run(
        ["fluidsynth", "-ni", "-q", "-g", "0.6", "-r", str(sr), "-F", str(wav_path), str(sf), str(midi_path)],
        check=True,
        capture_output=True,
    )


def make_fixture(name: str, out_dir: Path, **kwargs) -> tuple[Path, list[Segment]]:
    """Write <name>.mid, <name>.wav and <name>.json (ground truth) into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pm, segments = build_midi(PROGRESSIONS[name], **kwargs)
    mid, wav, gt = (out_dir / f"{name}{ext}" for ext in (".mid", ".wav", ".json"))
    pm.write(str(mid))
    if not wav.exists():
        render_wav(mid, wav)
    gt.write_text(json.dumps([{"start": s.start, "end": s.end, "chord": s.chord.mir_eval()} for s in segments]))
    return wav, segments


def beat_accuracy(beats: list[dict], segments: list[Segment], step: float = 0.01) -> float:
    """Fraction of chord-bearing ground-truth time where the estimated chord (root
    and quality) matches. `beats` are dicts from the harmony result."""
    total = correct = 0
    est_i = 0
    for seg in segments:
        if seg.chord.is_none:
            continue
        for t in np.arange(seg.start, seg.end, step):
            while est_i < len(beats) and beats[est_i]["end"] <= t:
                est_i += 1
            est = beats[est_i]["mir_eval"] if est_i < len(beats) and beats[est_i]["start"] <= t else "N"
            total += 1
            correct += est == seg.chord.mir_eval()
    return correct / total if total else 0.0
