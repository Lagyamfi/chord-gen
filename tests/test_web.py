import json

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from chordsheet.web.server import create_app  # noqa: E402

KEY = "a" * 64


def _sheet(transcriber):
    beat = {"start": 0.0, "end": 0.5, "chord": "C", "mir_eval": "C:maj", "roman": "I", "in_key": True}
    return {"source": "https://www.youtube.com/watch?v=abcdefghijk", "cache_key": KEY, "transcriber": transcriber,
            "key": "C major", "key_alternatives": [], "tempo": 120.0, "bars": [[beat]]}


@pytest.fixture
def client(tmp_path):
    d = tmp_path / "cache" / KEY
    d.mkdir(parents=True)
    sf.write(d / "audio.wav", np.zeros(4410), 44100)
    for system in ("chroma", "muscriptor-large"):
        (d / f"sheet.{system}.json").write_text(json.dumps(_sheet(system)))
    (tmp_path / "cache" / "not-a-song").mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,title,artist,url\nx,Test Song,Someone,https://www.youtube.com/watch?v=abcdefghijk\n")
    return TestClient(create_app(tmp_path / "cache", manifest))


def test_song_list_prefers_best_system_and_uses_titles(client):
    [song] = client.get("/api/songs").json()
    assert song["title"] == "Test Song" and song["artist"] == "Someone"
    assert song["systems"] == ["muscriptor-large", "chroma"]


def test_sheet_default_and_explicit_system(client):
    assert client.get(f"/api/songs/{KEY}/sheet").json()["transcriber"] == "muscriptor-large"
    assert client.get(f"/api/songs/{KEY}/sheet?system=chroma").json()["transcriber"] == "chroma"
    assert client.get(f"/api/songs/{KEY}/sheet?system=nope").status_code == 404


def test_audio_supports_range_and_rejects_bad_keys(client):
    r = client.get(f"/api/songs/{KEY}/audio", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert client.get("/api/songs/..%2F..%2Fetc/audio").status_code == 404
    assert client.get("/").status_code == 200
