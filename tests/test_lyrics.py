import json

import numpy as np
import pytest
import soundfile as sf

from chordsheet.lyrics import LyricsUnavailable, choose, find_lyrics, parse_lrc

LRC = "[00:55.10] \n[01:08.32] I am a walking contradiction\n[01:12.5]Hated\n[00:20.00] First line"


def test_parse_lrc_sorts_and_drops_empty_lines():
    assert parse_lrc(LRC) == [
        {"t": 20.0, "text": "First line"},
        {"t": 68.32, "text": "I am a walking contradiction"},
        {"t": 72.5, "text": "Hated"},
    ]


def test_choose_the_version_closest_in_length():
    results = [
        {"id": 1, "duration": 186, "syncedLyrics": LRC},
        {"id": 2, "duration": 245, "syncedLyrics": LRC},
        {"id": 3, "duration": 244, "syncedLyrics": None},  # plain lyrics only
    ]
    assert choose(results, 246.0)["id"] == 2
    assert choose(results, 300.0) is None  # nothing within 30 s: a different arrangement


def test_find_lyrics_falls_back_to_a_free_search_and_flags_drift(monkeypatch):
    monkeypatch.setattr("chordsheet.lyrics.time.sleep", lambda s: None)
    calls = []

    def search(params):
        calls.append(params)
        return [] if "track_name" in params else [{"id": 9, "trackName": "Hated", "artistName": "Zach Williams",
                                                   "duration": 240, "syncedLyrics": LRC}]

    found = find_lyrics("Hated (Live)", "Zach Williams", 246.0, search=search)
    assert calls[0] == {"track_name": "Hated", "artist_name": "Zach Williams"}  # version stripped
    assert found["found"] and found["drift"] and found["duration_gap"] == -6.0 and len(found["lines"]) == 3


pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from chordsheet.web.jobs import Jobs  # noqa: E402
from chordsheet.web.server import create_app  # noqa: E402

KEY = "b" * 64


def _client(tmp_path, find):
    d = tmp_path / KEY
    d.mkdir()
    sf.write(d / "audio.wav", np.zeros(44100 * 2), 44100)
    (d / "sheet.chroma.json").write_text(json.dumps({"source": "https://youtu.be/abcdefghijk", "bars": []}))
    (d / "meta.json").write_text(json.dumps({"title": "Hated", "artist": "Zach Williams"}))
    return TestClient(create_app(tmp_path, None, Jobs(tmp_path, lambda *a: ""), find_lyrics=find)), d


def test_lyrics_are_looked_up_once_then_cached_and_can_be_nudged(tmp_path):
    asked = []

    def find(title, artist, duration):
        asked.append((title, artist, round(duration)))
        return {"found": True, "checked": 0, "offset": 0.0, "lines": [{"t": 1.0, "text": "Hated"}]}

    client, d = _client(tmp_path, find)
    assert client.get(f"/api/songs/{KEY}/lyrics").json()["lines"][0]["text"] == "Hated"
    client.get(f"/api/songs/{KEY}/lyrics")
    assert asked == [("Hated", "Zach Williams", 2)]
    assert client.post(f"/api/songs/{KEY}/lyrics/offset", json={"offset": 0.5}).json() == {"offset": 0.5}
    assert client.get(f"/api/songs/{KEY}/lyrics").json()["offset"] == 0.5


def test_a_busy_lrclib_caches_nothing(tmp_path):
    def busy(*a):
        raise LyricsUnavailable("LRCLIB is busy (429)")

    client, d = _client(tmp_path, busy)
    r = client.get(f"/api/songs/{KEY}/lyrics").json()
    assert r["retry"] and not (d / "lyrics.json").exists()
