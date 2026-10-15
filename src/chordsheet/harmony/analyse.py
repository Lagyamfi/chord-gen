"""Layer 3 entry point: note events + beat grid -> chords per beat and key."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chordsheet.harmony.key import Key, estimate_keys, in_key, parse_key, roman
from chordsheet.harmony.profile import beat_profiles
from chordsheet.harmony.templates import QUALITIES, TRIADS, Chord, build_templates, score_beats
from chordsheet.harmony.viterbi import viterbi
from chordsheet.models import BeatGrid, NoteEvent

# A close call (shown dotted, with the runner-up): the chord's template score leads the
# runner-up's by under UNSURE_MARGIN, or by under SHORT_MARGIN when the chord lasts
# under SHORT_SECONDS (a quarter of short chords are wrong, against 2% of long ones).
# Tuned on the four ground-truth songs with scripts/tune_confidence.py: half the wrong
# chords marked, 7% of the right ones. Only 22 wrong chords, so treat as a first fit.
UNSURE_MARGIN = 0.02
SHORT_SECONDS = 2.0
SHORT_MARGIN = 0.10


@dataclass
class HarmonyResult:
    windows: list[tuple[float, float]]
    chords: list[Chord]
    bass_pc: list[int]
    keys: list[Key]  # ranked, best first (the set key first when overridden)
    key_set: bool = False  # True when the key came from the user, not detection
    margins: list[float] | None = None  # per beat: the chord's lead over its runner-up
    alts: list[Chord | None] | None = None  # per beat: the runner-up chord
    unsure: list[bool] | None = None  # per beat: its chord is a close call

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
                    **_confidence(m, a, u, key),
                }
                for (s, e), c, b, m, a, u in zip(
                    self.windows, self.chords, self.bass_pc,
                    self.margins or [None] * len(self.chords), self.alts or [None] * len(self.chords),
                    self.unsure or [False] * len(self.chords),
                )
            ],
        }


def _confidence(margin: float | None, alt: Chord | None, unsure: bool, key: Key) -> dict:
    if margin is None or alt is None:
        return {}
    return {"conf": round(margin, 3), "unsure": unsure, "alt": alt.name(), "alt_roman": roman(alt, key)}


def run_confidence(
    scores: np.ndarray, path: np.ndarray, chords: list[Chord], windows: list[tuple[float, float]], key: Key
) -> tuple[list[float], list[Chord | None], list[bool]]:
    """Per beat: how far its chord's score leads the best other chord, the runner-up,
    and whether that makes it a close call. All are taken over the whole run of beats
    on that chord (one beat is too noisy to judge). No-chord runs get none of this,
    and no-chord is never a runner-up.

    When major and minor on the same root score the same, the notes had no third,
    and "F or Fm?" is the wrong question: the runner-up is then the best chord on
    another root in the key (F-C alone is as much C as F; F alone fits a dozen chords,
    and only the in-key ones are worth asking about)."""
    n = len(path)
    margins: list[float] = [0.0] * n
    alts: list[Chord | None] = [None] * n
    unsure = [False] * n
    playable = np.array([not c.is_none for c in chords])
    start = 0
    for t in range(1, n + 1):
        if t < n and path[t] == path[start]:
            continue
        c = int(path[start])
        if not chords[c].is_none:
            mean = scores[start:t].mean(axis=0)
            others = np.where(playable, mean, -np.inf)
            others[c] = -np.inf
            a = int(others.argmax())
            margin = float(mean[c] - mean[a])
            if chords[a].root == chords[c].root and margin < 1e-6:
                others[[j for j, ch in enumerate(chords) if ch.root == chords[c].root or not in_key(ch, key)]] = -np.inf
                if np.isfinite(others.max()):
                    a = int(others.argmax())
            seconds = windows[t - 1][1] - windows[start][0]
            close = margin < UNSURE_MARGIN or (seconds < SHORT_SECONDS and margin < SHORT_MARGIN)
            margins[start:t] = [margin] * (t - start)
            alts[start:t] = [chords[a]] * (t - start)
            unsure[start:t] = [close] * (t - start)
        start = t
    return margins, alts, unsure


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
    margins, alts, unsure = run_confidence(scores, path, chords, windows, keys[0])
    return HarmonyResult(
        windows, [chords[i] for i in path], [int(b) for b in bass_pc], keys, key_set=key is not None,
        margins=margins, alts=alts, unsure=unsure,
    )
