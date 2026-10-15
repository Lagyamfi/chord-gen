from chordsheet.guitar.progressions import (
    as_loop,
    describe,
    label,
    library,
    line_numerals,
    loop_key,
)


def test_line_numerals_collapse_and_skip_no_chord():
    bars = [
        [{"roman": "-"}, {"roman": "I"}, {"roman": "I"}],
        [{"roman": "V"}, {"roman": "vi"}],
    ]
    assert line_numerals(bars) == ["I", "V", "vi"]


def test_as_loop_reduces_repeats_and_the_return_home():
    assert as_loop(["I", "IV", "I", "IV"]) == ["I", "IV"]
    assert as_loop(["I", "vi", "V", "IV", "I"]) == ["I", "vi", "V", "IV"]
    assert as_loop(["V7", "I"]) == ["V", "I"]  # extensions match on the triad


def test_rotations_share_a_key():
    assert loop_key(["vi", "IV", "I", "V"]) == loop_key(["I", "V", "vi", "IV"])
    assert loop_key(["I", "IV", "V"]) != loop_key(["I", "V", "IV"])  # order matters


def test_named_loops_and_where_the_song_starts_them():
    d = describe(["vi", "IV", "I", "V"])
    assert d["name"] == "Four-chord loop" and d["start"] == "vi"
    assert label(d) == "vi – IV – I – V · Four-chord loop, from vi"
    assert describe(["I", "V", "vi", "IV"])["start"] is None
    assert describe(["I", "IV", "I", "IV"])["name"] == "Two-chord vamp"
    assert describe(["I", "V", "II", "vi"])["name"] is None  # numerals only
    assert describe(["I"]) is None


def _sheet(*lines):
    """A tiny sheet: one section per line, each line one bar of numerals."""
    bars = [[{"roman": r} for r in line] for line in lines]
    letters = [{"label": chr(65 + i), "rep": i, "length": 1} for i in range(len(lines))]
    blocks = [
        {"title": f"S{i}", "lines": [[chr(65 + i), 1, [float(i)]]]}
        for i in range(len(lines))
    ]
    return {"bars": bars, "form": {"letters": letters}, "sections": {"blocks": blocks}}


def test_library_groups_the_same_loop_across_songs():
    songs = [
        {
            "id": "a",
            "title": "Song A",
            "sheet": _sheet(["I", "V", "vi", "IV"], ["I", "IV"]),
        },
        {"id": "b", "title": "Song B", "sheet": _sheet(["vi", "IV", "I", "V"])},
    ]
    lib = library(songs)
    four = lib[describe(["I", "V", "vi", "IV"])["key"]]
    assert [(o["title"], o["section"]) for o in four] == [
        ("Song A", "S0"),
        ("Song B", "S0"),
    ]
    assert len(lib) == 2
