from collections import Counter

from chordsheet.guitar.capo import suggest_capo
from chordsheet.guitar.chords import parse, transpose_name
from chordsheet.guitar.voicings import SHAPES, is_open_shape, voicing


def test_parse_and_transpose():
    assert parse("C#m").root == 1 and parse("C#m").suffix == "m"
    assert parse("G/B").bass == 11
    assert parse("N.C.") is None
    assert transpose_name("Ab", -1) == "G"
    assert transpose_name("Bbm7/F", 2) == "Cm7/G"
    assert transpose_name("N.C.", 3) == "N.C."


def test_curated_table_is_well_formed():
    assert len(SHAPES) >= 40
    for name, frets in SHAPES.items():
        assert parse(name) is not None, name
        assert len(frets) == 6 and set(frets) <= set("x0123456789"), name


def test_voicings():
    assert voicing("G").frets == (3, 2, 0, 0, 0, 3) and voicing("G").is_open
    assert voicing("D/F#").frets == (2, 0, 0, 2, 3, 2)
    assert voicing("Em/B").frets == voicing("Em").frets  # slash falls back to the plain shape
    bb = voicing("Bb")  # A-shape barre at fret 1
    assert bb.barre and bb.generated and bb.frets == (None, 1, 3, 3, 3, 1)
    fsharp_m = voicing("F#m")  # E-shape barre at fret 2
    assert fsharp_m.frets == (2, 4, 4, 2, 2, 2)
    assert voicing("N.C.") is None
    assert is_open_shape("F") and is_open_shape("Fmaj7") and not is_open_shape("Bb")


def test_capo_for_ab_song_is_1():
    # There Was Jesus studio key: Ab Cm Db Fm Eb -> capo 1 gives G Bm C Em D
    song = Counter({"Ab": 40, "Cm": 10, "Db": 20, "Fm": 15, "Eb": 15})
    choice = suggest_capo(song)
    assert choice.fret == 1 and choice.open_fraction > 0.85 and choice.open_fraction_no_capo == 0


def test_c_major_song_needs_no_capo():
    # C F G Am: with the small F everything is open; capo 5 (G C D Em) must not win
    assert suggest_capo(Counter({"C": 10, "F": 10, "G": 10, "Am": 10})).fret == 0


def test_bb_song_gets_capo_3():
    # Bb F C Gm -> capo 3: G D A Em
    assert suggest_capo(Counter({"Bb": 10, "F": 10, "C": 10, "Gm": 10})).fret == 3


def test_no_capo_when_already_open():
    choice = suggest_capo(Counter({"G": 4, "C": 4, "D": 4, "Em": 4}))
    assert choice.fret == 0 and choice.open_fraction == 1.0
