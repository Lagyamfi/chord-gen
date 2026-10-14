import numpy as np

from chordsheet.guitar.sections import apply_names, guess_sections, song_sections


def lines_of(*labels):
    return [{"label": l, "start": 4 * i, "length": 4} for i, l in enumerate(labels)]


def bars_for(n_lines):
    return [[{"start": 2.0 * i, "end": 2.0 * (i + 1)}] for i in range(4 * n_lines)]


def test_loud_repeated_lines_are_chorus_quiet_are_verse():
    lines = lines_of("-", "A", "A", "B", "C", "A", "A", "B", "C", "C")
    energy = np.array([-3, -1, -1, 1.5, 1.5, -1, -1, 1.5, 1.5, 1.4])
    names = guess_sections(lines, energy)
    assert names[0] == "Intro"
    assert names[1:3] == ["Verse", "Verse"]
    assert names[3:5] == ["Chorus", "Chorus"]


def test_chorus_twice_in_a_row_folds_and_blocks_group():
    lines = lines_of("A", "A", "E", "F", "E", "F", "A", "A", "E", "F")
    names = ["Verse"] * 2 + ["Chorus"] * 4 + ["Verse"] * 2 + ["Chorus"] * 2
    sec = song_sections(lines, bars_for(len(lines)), names)
    assert [(r["name"], r["reps"]) for r in sec["runs"]] == [("Verse", 1), ("Chorus", 2), ("Verse", 1), ("Chorus", 1)]
    chorus = next(b for b in sec["blocks"] if b["name"] == "Chorus")
    assert [l[:2] for l in chorus["lines"]] == [["E", 1], ["F", 1]] and chorus["count"] == 3
    assert chorus["title"] == "Chorus" and chorus["lines"][1][2] == [24.0, 72.0]  # where F starts in each run
    verse = next(b for b in sec["blocks"] if b["name"] == "Verse")
    assert [l[:2] for l in verse["lines"]] == [["A", 2]] and verse["count"] == 2  # line A twice, in two verses


def test_rename_applies_by_time_and_different_lines_get_numbered_titles():
    lines = lines_of("A", "B", "C", "D")
    bars = bars_for(4)
    guessed = ["Verse", "Verse", "Bridge", "Chorus"]
    names = apply_names(lines, bars, guessed, {"16.4": "Chorus"})  # the Bridge starts at 16.0
    assert names == ["Verse", "Verse", "Chorus", "Chorus"]
    sec = song_sections(lines, bars, ["Verse", "Verse", "Chorus", "Verse"])
    assert [b["title"] for b in sec["blocks"]] == ["Verse", "Chorus", "Verse (2)"]


def test_new_section_starts_mid_way():
    lines = lines_of("E", "F", "B", "B")
    guessed = ["Chorus"] * 4  # the guess ran the chorus into the verse
    names = apply_names(lines, bars_for(4), guessed, {"16.0": "Verse"})  # line 3 starts at 16 s
    assert names == ["Chorus", "Chorus", "Verse", "Verse"]
