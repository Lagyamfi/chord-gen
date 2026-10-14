import numpy as np
import pytest

from chordsheet.evaluation.chart import parse_chart, parse_chord

pytest.importorskip("mir_eval")
from chordsheet.evaluation.align import align, chord_template  # noqa: E402

CHART = """
Hated chords
Capo 2

[Intro]
G  D/F#  Em  C

[Verse 1]
G                D/F#
Some lyric line here
Em             C
Another lyric line

[Chorus]
C        G
Chorus words
D        Em
More chorus x2

[Verse 2]
Lyrics only, no chords

[Chorus]
"""


@pytest.mark.parametrize(
    "token,label",
    [
        ("G", "G:maj"), ("Em", "E:min"), ("D/F#", "D:maj/3"), ("Bbmaj7", "Bb:maj7"),
        ("F#m7", "F#:min7"), ("Asus4", "A:sus4"), ("Esus2", "E:sus2"), ("E2", "E:sus2"),
        ("Cadd9", "C:maj(9)"), ("C(add9)", "C:maj(9)"), ("A7sus4", "A:sus4(b7)"), ("Bdim", "B:dim"),
        ("G/B", "G:maj/3"), ("C/G", "C:maj/5"), ("Am7/G", "A:min7/b7"),
    ],
)
def test_parse_chord(token, label):
    assert parse_chord(token).label == label


def test_no_chord_and_anchored_section_not_merged():
    assert parse_chord("N.C.").label == "N" and parse_chord("NC").transpose(3).label == "N"
    chart = parse_chart("[Chorus]\nF C\n[Verse 2] @0:56\nC\n[Break]\nN.C.\n")
    seq = chart.sequence()
    assert [(it.chord.label, it.section) for it in seq] == [
        ("F:maj", "Chorus"), ("C:maj", "Chorus"), ("C:maj", "Verse 2"), ("N", "Break")
    ]
    assert chart.anchors() == [(2, 56.0, "Verse 2")]


@pytest.mark.parametrize("token", ["Some", "lyric", "Am I", "x2", "Go"])
def test_non_chords(token):
    assert parse_chord(token) is None


def test_parse_chart_capo_repeats_and_reuse():
    chart = parse_chart(CHART)
    assert chart.capo == 2
    names = [s.name for s in chart.sections]
    assert names == ["Intro", "Verse 1", "Chorus", "Verse 2", "Chorus"]
    verse2, chorus2 = chart.sections[3], chart.sections[4]
    assert verse2.reused and chorus2.reused
    # "More chorus x2" is a lyric line, so the chord line above it is not repeated
    assert [c.label for c in chart.sections[2].chords] == ["C:maj", "G:maj", "D:maj", "E:min"]
    seq = chart.sequence()
    # capo 2: G shapes sound as A
    assert seq[0][0].label == "A:maj" and seq[1][0].label == "E:maj/3"


def test_align_recovers_boundaries_and_key():
    # audio: 2 beats silence, then A E F#m D, 4 beats each (G D Em C shapes with capo 2)
    chart = parse_chart("Capo 2\n[Verse]\nG D Em C\n")
    seq = chart.sequence()
    truth = [it.chord for it in seq]
    profiles = np.vstack([np.zeros((2, 12))] + [np.tile(chord_template(c), (4, 1)) for c in truth])
    bass = np.array([-1, -1] + [c.root for c in truth for _ in range(4)])
    result = align(seq, profiles, bass)
    assert result.transpose == 0
    assert result.labels == ["N", "N"] + [c.label for c in truth for _ in range(4)]

    # the same chart without the capo line is 2 semitones low; the search finds +2
    seq_nocapo = parse_chart("[Verse]\nG D Em C\n").sequence()
    assert align(seq_nocapo, profiles, bass).transpose == 2


def test_align_loops_a_repeated_section():
    # chart: verse C F once, then chorus G Am; audio plays the verse twice
    seq = parse_chart("[Verse]\nC F\n[Chorus]\nG Am\n").sequence()
    chords = [parse_chord(t) for t in ["C", "F", "C", "F", "G", "Am"]]
    profiles = np.vstack([np.tile(chord_template(c), (4, 1)) for c in chords])
    bass = np.array([c.root for c in chords for _ in range(4)])
    result = align(seq, profiles, bass, transpose=0, allow_loops=True)
    assert result.labels == [c.label for c in chords for _ in range(4)]
    assert [name for name, _ in result.loops] == ["Verse"]
    # without loops the verse has to stretch, so some beats are wrong
    assert align(seq, profiles, bass, transpose=0).labels != result.labels


def test_anchor_forces_section_start():
    chart = parse_chart("[Intro] @0:00\nC F\n[Verse] @0:04\nC F\n")
    assert [a[1] for a in chart.anchors()] == [0.0, 4.0]
    seq = chart.sequence()
    # audio: C for 6 beats then F for 2; without the anchor the intro's C covers all six
    chords = [parse_chord(t) for t in ["C"] * 6 + ["F"] * 2]
    profiles = np.array([chord_template(c) for c in chords])
    bass = np.array([c.root for c in chords])
    result = align(seq, profiles, bass, transpose=0, anchors=[(2, 4)])
    assert result.sections[3] == "Intro" and result.sections[4] == "Verse"
