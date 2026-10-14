import numpy as np

from chordsheet.harmony.key import Key, estimate_keys, in_key, roman
from chordsheet.harmony.profile import beat_profiles
from chordsheet.harmony.templates import Chord, build_templates, score_beats
from chordsheet.harmony.viterbi import viterbi
from chordsheet.models import NoteEvent


def notes_for(chord_pitches, start, end, instrument="piano"):
    return [NoteEvent(p, start, end, 100.0, instrument) for p in chord_pitches]


def test_profile_and_bass():
    notes = notes_for([60, 64, 67], 0, 1) + [NoteEvent(40, 0, 1, 100, "bass")]  # C major over E
    profiles, bass = beat_profiles(notes, [(0, 1), (1, 2)])
    assert np.isclose(np.linalg.norm(profiles[0]), 1)
    assert set(np.flatnonzero(profiles[0])) == {0, 4, 7}
    assert bass[0] == 4 and bass[1] == -1
    assert not profiles[1].any()


def test_drums_ignored():
    profiles, _ = beat_profiles([NoteEvent(36, 0, 1, 100, "drums", is_drum=True)], [(0, 1)])
    assert not profiles.any()


def test_template_scoring_picks_triad_and_slash():
    chords, templates = build_templates()
    notes = notes_for([57, 60, 64], 0, 1) + [NoteEvent(45, 0, 1, 100, "bass")]  # A minor
    profiles, bass = beat_profiles(notes, [(0, 1)])
    best = chords[int(score_beats(profiles, bass, chords, templates)[0].argmax())]
    assert best == Chord(9, "min")
    assert Chord(0, "maj").name(bass_pc=4) == "C/E"


def test_silence_is_no_chord():
    chords, templates = build_templates()
    scores = score_beats(np.zeros((1, 12)), np.array([-1]), chords, templates)
    assert chords[int(scores[0].argmax())].is_none


def test_viterbi_removes_flicker():
    # state 0 dominant, with a one-beat blip of state 1
    scores = np.full((8, 3), 0.2)
    scores[:, 0] = 0.9
    scores[4] = [0.8, 0.95, 0.2]
    assert list(viterbi(scores)) == [0] * 8
    # a sustained change is kept
    scores[4:] = [0.2, 0.95, 0.2]
    assert list(viterbi(scores)) == [0] * 4 + [1] * 4


def test_key_and_roman():
    chords, templates = build_templates()
    prog = [(7, "maj"), (2, "maj"), (4, "min"), (0, "maj")]  # G D Em C
    profiles = np.array([templates[chords.index(Chord(r, q))] for r, q in prog] * 4)
    key = estimate_keys(profiles)[0]
    assert key.name == "G major"
    assert [roman(Chord(r, q), key) for r, q in prog] == ["I", "V", "vi", "IV"]
    assert roman(Chord(4, "min7"), key) == "vi7"
    assert roman(Chord(2, "7"), key) == "V7"
    assert in_key(Chord(0, "maj"), key) and not in_key(Chord(5, "maj"), key)  # F is bVII in G
    assert roman(Chord(5, "maj"), key) == "bVII"


def test_minor_key_dominant_is_diatonic():
    key = Key(9, "minor", 1.0)
    assert in_key(Chord(4, "maj"), key) and roman(Chord(4, "maj"), key) == "V"
    assert roman(Chord(0, "maj"), key) == "bIII"


def test_bar_phase_from_chord_changes():
    from chordsheet.guitar.bars import best_bar_phase

    # pickup beat, then one chord per 4-beat bar starting at beat index 1
    chords = ["N"] + ["G"] * 4 + ["D"] * 4 + ["Em"] * 4
    assert best_bar_phase(chords) == 1


def test_snap_to_beats():
    from chordsheet.pipeline import snap_to_beats

    assert snap_to_beats([0.02, 1.98, 2.01], [0.0, 0.5, 1.0, 1.5, 2.0]) == [0.0, 2.0]


def test_parse_key():
    import pytest

    from chordsheet.harmony.key import parse_key

    assert parse_key("Bb") == (10, "major")
    assert parse_key("Bb major") == (10, "major")
    assert parse_key("G minor") == (7, "minor")
    assert parse_key("Em") == (4, "minor")
    assert parse_key("f#m") == (6, "minor")
    with pytest.raises(ValueError):
        parse_key("H major")


def test_key_override_changes_numerals_not_chords():
    from chordsheet.harmony.analyse import analyse_harmony
    from chordsheet.models import BeatGrid

    # Bb F C Gm, one beat each: detection says F major (C major chord is diatonic to F)
    notes = []
    for i, pcs in enumerate([(58, 62, 65), (53, 57, 60), (60, 64, 67), (55, 58, 62)] * 4):
        notes += notes_for(pcs, i, i + 1)
    grid = BeatGrid([float(i) for i in range(16)], [0.0, 4.0, 8.0, 12.0], 60.0, 16.0)
    detected = analyse_harmony(notes, grid, self_transition=0.5).to_dict()
    forced = analyse_harmony(notes, grid, self_transition=0.5, key="Bb").to_dict()
    assert forced["key"] == "Bb major" and forced["key_source"] == "set"
    assert detected["key_source"] == "detected"
    assert [b["chord"] for b in forced["beats"]] == [b["chord"] for b in detected["beats"]]
    assert [b["roman"] for b in forced["beats"][:4]] == ["I", "V", "II", "vi"]
    assert [b["in_key"] for b in forced["beats"][:4]] == [True, True, False, True]
