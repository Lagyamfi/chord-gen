from chordsheet.evaluation.synthetic import beat_accuracy, make_fixture
from chordsheet.pipeline import analyse


def test_g_d_em_c_chroma_fallback(fixtures_dir, tmp_path):
    wav, segments = make_fixture("g_d_em_c", fixtures_dir)
    sheet = analyse(str(wav), transcriber="chroma", sevenths=False, cache_root=tmp_path)
    beats = [b for bar in sheet["bars"] for b in bar]

    assert sheet["key"] == "G major"
    # collapse beat chords into the change sequence, ignoring no-chord
    seq = []
    for b in beats:
        if b["mir_eval"] != "N" and (not seq or seq[-1] != b["mir_eval"]):
            seq.append(b["mir_eval"])
    assert seq == ["G:maj", "D:maj", "E:min", "C:maj"] * 2
    assert beat_accuracy(beats, segments) >= 0.9


def test_cache_hit(fixtures_dir, tmp_path):
    wav, _ = make_fixture("g_d_em_c", fixtures_dir)
    first = analyse(str(wav), cache_root=tmp_path)
    notes_file = next(tmp_path.glob("*/notes.chroma.json"))
    mtime = notes_file.stat().st_mtime
    second = analyse(str(wav), cache_root=tmp_path)
    assert notes_file.stat().st_mtime == mtime
    assert first["bars"] == second["bars"]
