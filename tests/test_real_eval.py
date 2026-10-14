import numpy as np
import pytest

from chordsheet.evaluation.real import parse_system, score, sheet_intervals, write_lab

pytest.importorskip("mir_eval")


def _sheet(chords):
    beats = [{"start": i * 0.5, "end": (i + 1) * 0.5, "mir_eval": c} for i, c in enumerate(chords)]
    return {"bars": [beats[i : i + 4] for i in range(0, len(beats), 4)]}


def test_sheet_intervals_merges_repeats():
    intervals, labels = sheet_intervals(_sheet(["G:maj"] * 4 + ["D:maj"] * 2 + ["E:min"] * 2))
    assert labels == ["G:maj", "D:maj", "E:min"]
    assert np.allclose(intervals, [[0, 2], [2, 3], [3, 4]])


def test_perfect_and_wrong_scores(tmp_path):
    intervals, labels = sheet_intervals(_sheet(["G:maj"] * 4 + ["E:min"] * 4))
    lab = tmp_path / "ref.lab"
    write_lab(lab, intervals, labels)
    assert score(lab, intervals, labels) == {"root": 1.0, "majmin": 1.0, "sevenths": 1.0}
    # G major vs G minor: right root, wrong quality
    s = score(lab, intervals, ["G:min", "E:min"])
    assert s["root"] == 1.0 and s["majmin"] == pytest.approx(0.5)


def test_parse_system():
    assert parse_system("chroma") == ("chroma", "")
    assert parse_system("muscriptor-large") == ("muscriptor", "large")
    assert parse_system("muscriptor") == ("muscriptor", "medium")
