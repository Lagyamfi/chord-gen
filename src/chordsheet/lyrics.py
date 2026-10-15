"""Line-timed lyrics from LRCLIB (https://lrclib.net), a free open lyrics database.

A song is looked up once by title and artist; among the versions with line
timings, the one whose length is closest to our recording wins (live and studio
versions differ). The result, or the fact that nothing was found, is cached as
lyrics.json in the song's cache folder. Lyrics are copyrighted: they live in the
cache only, never in git."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://lrclib.net/api/search"
USER_AGENT = (
    "ChordSheet/0.1 (personal learning tool; https://github.com/Lagyamfi/chord-gen)"
)
MAX_LENGTH_GAP = 30.0  # s: a version further off than this is a different arrangement
DRIFT_WARNING = 4.0  # s: closer than this and the timings should line up
RETRY_NONE_AFTER = 7 * 24 * 3600  # s: look again for songs that had no lyrics


class LyricsUnavailable(Exception):
    """LRCLIB could not be reached or asked us to slow down: try again later, cache nothing."""


_LRC = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")


def parse_lrc(text: str) -> list[dict]:
    """'[01:08.32] I am a walking...' lines -> [{t, text}], sorted, empty lines kept
    out (they mark instrumental gaps, which the chart already shows)."""
    lines = []
    for raw in text.splitlines():
        m = _LRC.match(raw.strip())
        if m and m.group(3).strip():
            lines.append(
                {
                    "t": round(int(m.group(1)) * 60 + float(m.group(2)), 2),
                    "text": m.group(3).strip(),
                }
            )
    return sorted(lines, key=lambda l: l["t"])


def _plain(title: str) -> str:
    """'There Was Jesus (Live)' -> 'There Was Jesus': LRCLIB titles rarely carry the version."""
    return re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip() or title


def _search(params: dict) -> list[dict]:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (429, 500, 502, 503, 504):
            raise LyricsUnavailable(f"LRCLIB is busy ({e.code})") from None
        raise
    except (urllib.error.URLError, TimeoutError) as e:
        raise LyricsUnavailable(f"couldn't reach LRCLIB ({e})") from None
    return data if isinstance(data, list) else []


def choose(results: list[dict], duration: float) -> dict | None:
    """The synced version closest in length to the recording, if close enough."""
    synced = [r for r in results if r.get("syncedLyrics") and r.get("duration")]
    if not synced:
        return None
    best = min(synced, key=lambda r: abs(r["duration"] - duration))
    return best if abs(best["duration"] - duration) <= MAX_LENGTH_GAP else None


def find_lyrics(title: str, artist: str, duration: float, search=_search) -> dict:
    """{found, lines, source, id, name, duration_gap, drift, checked} for a song."""
    found = None
    for params in (
        {"track_name": _plain(title), "artist_name": artist},
        {"q": f"{_plain(title)} {artist}".strip()},
    ):
        found = choose(search(params), duration)
        if found:
            break
        time.sleep(0.5)  # LRCLIB rate-limits bursts
    out = {"found": bool(found), "checked": time.time(), "offset": 0.0}
    if found:
        gap = round(found["duration"] - duration, 1)
        out |= {
            "source": "LRCLIB",
            "id": found.get("id"),
            "name": f"{found.get('trackName', '')} · {found.get('artistName', '')}",
            "duration_gap": gap,
            "drift": abs(gap) > DRIFT_WARNING,
            "lines": parse_lrc(found["syncedLyrics"]),
        }
    return out


def stale(cached: dict) -> bool:
    """A 'nothing found' answer is worth asking again after a week."""
    return (
        not cached.get("found")
        and time.time() - cached.get("checked", 0) > RETRY_NONE_AFTER
    )
