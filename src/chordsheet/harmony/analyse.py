"""Layer 3 entry point: note events + beat grid -> chords per beat and key."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chordsheet.harmony.key import Key, estimate_keys, in_key, parse_key, roman
from chordsheet.harmony.profile import beat_profiles
from chordsheet.harmony.templates import QUALITIES, TRIADS, Chord, build_templates, score_beats
from chordsheet.harmony.viterbi import viterbi
from chordsheet.models import BeatGrid, NoteEvent


@dataclass
class HarmonyResult:
    windows: list[tuple[float, float]]
    chords: list[Chord]
    bass_pc: list[int]
    keys: list[Key]  # ranked, best first (the set key first when overridden)
    key_set: bool = False  # True when the key came from the user, not detection

    def to_dict(self) -> dict:
        key = self.keys[0]
        return {
            "key": key.name,
            "key_source": "set" if self.key_set else "detected",
            "key_alternatives": [{"key": k.name, "score": round(k.score, 3)} for k in self.keys[:3]],
            "beats": [
                {
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "chord": c.name(b),
                    "mir_eval": c.mir_eval(),
                    "roman": roman(c, key),
                    "in_key": in_key(c, key),
                }
                for (s, e), c, b in zip(self.windows, self.chords, self.bass_pc)
            ],
        }


def analyse_harmony(
    notes: list[NoteEvent],
    grid: BeatGrid,
    self_transition: float = 0.85,
    rich: bool = False,
    key: str | None = None,
) -> HarmonyResult:
    """`key` (e.g. "Bb", "Em") overrides key detection. It only changes the Roman
    numerals and the out-of-key flags; the chords themselves never depend on it."""
    windows = grid.windows()
    profiles, bass_pc = beat_profiles(notes, windows)
    # Plain major/minor by default: on real worship mixes the sus/7th templates mostly
    # mislabel open voicings (F with a quiet third read as Csus4), and triads are the
    # playable shapes a learner wants first. `rich` adds 7ths, sus4 and dim.
    chords, templates = build_templates(tuple(QUALITIES) if rich else TRIADS)
    scores = score_beats(profiles, bass_pc, chords, templates)
    path = viterbi(scores, self_transition=self_transition)
    keys = estimate_keys(profiles)
    if key is not None:
        tonic, mode = parse_key(key)
        chosen = next(k for k in keys if k.tonic == tonic and k.mode == mode)
        keys = [chosen, *(k for k in keys if k is not chosen)]
    return HarmonyResult(
        windows, [chords[i] for i in path], [int(b) for b in bass_pc], keys, key_set=key is not None
    )
