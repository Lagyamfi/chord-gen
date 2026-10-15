import json

import numpy as np
import pytest
import soundfile as sf

from chordsheet.ingest.youtube import YouTubeError, cache_key, clean_title, explain, video_id

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from chordsheet.web.jobs import Jobs  # noqa: E402
from chordsheet.web.server import create_app  # noqa: E402

URL = "https://www.youtube.com/watch?v=abcdefghijk"


def test_any_link_to_a_video_is_the_same_song():
    links = [URL, "https://youtu.be/abcdefghijk?si=x", "https://music.youtube.com/watch?v=abcdefghijk&list=RD1",
             "https://www.youtube.com/watch?list=PL1&v=abcdefghijk", "https://www.youtube.com/shorts/abcdefghijk"]
    assert {video_id(u) for u in links} == {"abcdefghijk"}
    assert len({cache_key(u) for u in links}) == 1
    assert video_id("https://example.com/song.mp3") is None


def test_titles_lose_the_noise_but_keep_versions():
    assert clean_title("Josiah Queen - Hated (Official Audio)", "Josiah Queen - Topic") == ("Hated", "Josiah Queen")
    assert clean_title("Hated [Official Lyric Video]", "JosiahQueenVEVO") == ("Hated", "JosiahQueen")
    assert clean_title("There Was Jesus (Live)", "Zach Williams") == ("There Was Jesus (Live)", "Zach Williams")
    assert clean_title("Someone - A Song", "") == ("A Song", "Someone")
    assert clean_title("Josiah Queen- I am Barabbas", "Josiah Queen") == ("I am Barabbas", "Josiah Queen")
    assert clean_title("Take It All - Hillsong Worship", "Hillsong Worship") == ("Take It All", "Hillsong Worship")
    assert clean_title("Well - Done", "Someone") == ("Well - Done", "Someone")  # a dash in the name itself


def test_download_errors_in_plain_words():
    assert "age-restricted" in explain("ERROR: [youtube] x: Sign in to confirm your age.")
    assert "private" in explain("ERROR: [youtube] x: Private video. Sign in")
    assert explain("ERROR: something new\n").endswith("(something new).")


def _write_song(cache_root, key, title=None):
    d = cache_root / key
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "audio.wav", np.zeros(4410), 44100)
    beat = {"start": 0.0, "end": 0.5, "chord": "C", "mir_eval": "C:maj", "roman": "I", "in_key": True}
    sheet = {"source": URL, "cache_key": key, "transcriber": "chroma", "key": "C major", "key_alternatives": [],
             "tempo": 120.0, "bars": [[beat]]}
    (d / "sheet.chroma.json").write_text(json.dumps(sheet))
    if title:
        (d / "meta.json").write_text(json.dumps({"title": title, "artist": "Someone"}))


def test_add_runs_in_the_background_and_the_song_appears_with_its_title(tmp_path):
    cache = tmp_path / "cache"
    seen = []

    def fake(url, root, progress):
        for stage in ("info", "download", "transcribe", "beats", "chords"):
            progress(stage)
            seen.append(stage)
        _write_song(root, cache_key(url), title="A Song")
        return cache_key(url)

    jobs = Jobs(cache, fake)
    client = TestClient(create_app(cache, None, jobs))
    job = client.post("/api/add", json={"url": URL}).json()
    assert job["status"] in ("queued", "running", "done")
    jobs.wait()
    [done] = client.get("/api/jobs").json()
    assert done["status"] == "done" and done["song"] == cache_key(URL)
    assert seen == ["info", "download", "transcribe", "beats", "chords"]
    assert [s["title"] for s in client.get("/api/songs").json()] == ["A Song"]

    # the same video again: already in the library, nothing re-runs
    again = client.post("/api/add", json={"url": "https://youtu.be/abcdefghijk"}).json()
    assert again["existing"] and again["status"] == "done" and len(seen) == 5


def test_add_rejects_non_links_and_reports_failures(tmp_path):
    def broken(url, root, progress):
        progress("download")
        raise YouTubeError("This video is private.")

    jobs = Jobs(tmp_path, broken)
    client = TestClient(create_app(tmp_path, None, jobs))
    assert client.post("/api/add", json={"url": "not a link"}).status_code == 422
    job = client.post("/api/add", json={"url": URL}).json()
    jobs.wait()
    [failed] = client.get("/api/jobs").json()
    assert failed["status"] == "failed" and failed["error"] == "This video is private."
    client.delete(f"/api/jobs/{job['id']}")
    assert client.get("/api/jobs").json() == []


def test_search_marks_official_versions_and_songs_already_in_the_library(tmp_path, monkeypatch):
    from chordsheet.ingest import youtube

    entries = [
        {"id": "aaaaaaaaaaa", "title": "Zach Williams - Hated (Lyric Video)", "channel": "Fan Channel", "duration": 187},
        {"id": "bbbbbbbbbbb", "title": "Zach Williams - Hated (Official Audio)", "channel": "Zach Williams", "duration": 188},
        {"id": "ccccccccccc", "title": "Hated - Zach Williams | Karaoke", "channel": "Zach Williams", "duration": 190},
        {"id": "ddddddddddd", "title": "Take It All - Live In Miami - Hillsong UNITED", "channel": "Hillsong UNITED",
         "duration": 255},
        {"id": "UCchannel", "title": "Zach Williams", "channel": "Zach Williams"},  # a channel, not a video
    ]
    monkeypatch.setattr(youtube, "_run", lambda args: json.dumps({"entries": entries}))
    results = youtube.search("zach williams hated")
    assert [(r["title"], r["artist"], r["official"]) for r in results] == [
        ("Hated", "Zach Williams", True),
        ("Take It All - Live In Miami", "Hillsong UNITED", True),
        ("Hated", "Zach Williams", False),  # a fan upload: the channel is someone else
        ("Hated - Karaoke", "Zach Williams", False),  # karaoke on the artist's channel, and it says so
    ]

    cache = tmp_path / "cache"
    _write_song(cache, youtube.cache_key("https://youtu.be/bbbbbbbbbbb"))
    client = TestClient(create_app(cache, None, Jobs(cache, lambda *a: "")))
    found = client.get("/api/search", params={"q": "hated"}).json()
    assert [r["in_library"] for r in found] == [True, False, False, False]
