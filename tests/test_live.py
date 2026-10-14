import soundfile as sf

from chordsheet.evaluation.synthetic import beat_accuracy, make_fixture
from chordsheet.live import LiveSession


def test_live_session_matches_whole_file(fixtures_dir, tmp_path):
    """Feeding a song in small blocks, transcribed in 8 s chunks, gives the same
    progression as analysing the whole file."""
    wav, segments = make_fixture("g_d_em_c", fixtures_dir)
    audio, sr = sf.read(wav, dtype="float32")
    session = LiveSession(sr, transcriber="chroma", chunk_s=8.0, overlap_s=1.0, cache_root=tmp_path)

    updates = []
    for i in range(0, len(audio), 2048):  # audio-callback-sized blocks
        sheet = session.feed(audio[i : i + 2048])
        if sheet:
            updates.append(sheet)
    final = session.finish() or updates[-1]

    assert len(updates) == int(len(audio) / sr // 8)  # one update per full chunk
    beats = [b for bar in final["bars"] for b in bar]
    seq = []
    for b in beats:
        if b["mir_eval"] != "N" and (not seq or seq[-1] != b["mir_eval"]):
            seq.append(b["mir_eval"])
    assert seq == ["G:maj", "D:maj", "E:min", "C:maj"] * 2
    assert beat_accuracy(beats, segments) >= 0.9
    assert (tmp_path / final["cache_key"] / "sheet.chroma.json").exists()  # visible to the player
