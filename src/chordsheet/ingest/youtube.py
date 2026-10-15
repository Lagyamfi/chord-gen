from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from chordsheet.cache import DEFAULT_ROOT, Cache, sha256_text
from chordsheet.ingest.file import decode_to_wav

_YT = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:[^#\s]*&)?v=|shorts/|live/)|music\.youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
)


class YouTubeError(Exception):
    """A download problem, worded for the player."""


def is_youtube_url(source: str) -> bool:
    return bool(_YT.search(source))


def video_id(url: str) -> str | None:
    match = _YT.search(url)
    return match.group(1) if match else None


def cache_key(url: str) -> str:
    """The song's cache key: the video id, so any link to the same video is the same song."""
    vid = video_id(url)
    if not vid:
        raise ValueError(f"not a YouTube URL: {url}")
    return sha256_text(f"youtube:{vid}")


# yt-dlp's error text -> what to tell Lawrence
_REASONS = [
    ("confirm your age", "This video is age-restricted, so it can't be downloaded."),
    ("private video", "This video is private."),
    ("members-only", "This video is for channel members only."),
    ("not available in your country", "This video isn't available in your country."),
    ("video unavailable", "This video is unavailable (removed, or the link is wrong)."),
    ("premieres in", "This video hasn't premiered yet."),
    ("live event", "This is a live stream that hasn't finished."),
    (
        "unable to download webpage",
        "Couldn't reach YouTube. Check the internet connection.",
    ),
    ("failed to resolve", "Couldn't reach YouTube. Check the internet connection."),
]


def explain(stderr: str) -> str:
    low = stderr.lower()
    for needle, reason in _REASONS:
        if needle in low:
            return reason
    last = next((l for l in reversed(stderr.strip().splitlines()) if l.strip()), "")
    return f"Couldn't download this video ({last.removeprefix('ERROR: ')[:160] or 'yt-dlp failed'})."


def _run(args: list[str]) -> str:
    try:
        r = subprocess.run(
            ["yt-dlp", "--no-playlist", *args], capture_output=True, text=True
        )
    except FileNotFoundError:
        raise YouTubeError("yt-dlp isn't installed (brew install yt-dlp).") from None
    if r.returncode != 0:
        raise YouTubeError(explain(r.stderr))
    return r.stdout


_NOISE = re.compile(
    r"\s*[\(\[][^)\]]*\b(official|audio|video|lyrics?|visuali[sz]er|hd|4k|hq|remaster(ed)?)\b[^)\]]*[\)\]]",
    re.I,
)


# the same noise unbracketed at the end: "... Official Video"
_TAIL = re.compile(r"[\s|-]*\bofficial( music| lyric)? (video|audio)\s*$", re.I)


def clean_title(title: str, artist: str) -> tuple[str, str]:
    """('Josiah Queen - Hated (Official Audio)', 'Josiah Queen - Topic') -> ('Hated', 'Josiah Queen').
    Keeps parts of the name that tell versions apart, like '(Live)'."""
    artist = re.sub(r"\s*(- Topic|VEVO)$", "", artist or "").strip()
    title = _NOISE.sub("", title or "").strip()
    title = _TAIL.sub("", title).strip()
    parts = _DASH.split(title, maxsplit=1)
    if len(parts) == 2:
        same = lambda x: re.sub(r"\W", "", x).casefold() == re.sub(r"\W", "", artist).casefold()
        left, right = (p.strip() for p in parts)
        if not artist:
            artist, title = left, right
        elif same(left):
            title = right
        elif same(right):
            title = left
    return title or "Untitled", artist


def fetch_info(url: str) -> dict:
    """{title, artist, duration (s), url} for a video, without downloading it."""
    info = json.loads(_run(["-J", "--skip-download", url]))
    title, artist = clean_title(
        info.get("track") or info.get("title", ""),
        info.get("artist")
        or info.get("creator")
        or info.get("channel")
        or info.get("uploader")
        or "",
    )
    return {
        "title": title,
        "artist": artist.split(",")[0].strip() if info.get("artist") else artist,
        "duration": info.get("duration") or 0,
        "url": url,
    }


def ingest_youtube(url: str, cache_root: Path = DEFAULT_ROOT) -> tuple[Cache, Path]:
    """Download audio with yt-dlp; the video id is the cache key."""
    cache = Cache(cache_key(url), cache_root)
    wav = cache.path("audio.wav")
    if not wav.exists():
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dl.%(ext)s"
            _run(["-q", "-x", "--audio-format", "wav", "-o", str(out), url])
            decode_to_wav(Path(tmp) / "dl.wav", wav)
    return cache, wav


_OFFICIAL = re.compile(r"\bofficial\b", re.I)
_NOT_ORIGINAL = re.compile(r"\b(karaoke|cover|instrumental|reaction|tutorial|lesson|how to play|backing track|8d|slowed|sped up|nightcore)\b", re.I)


_DASH = re.compile(r"\s*(?:[-–—]|\|+)\s+|\s+(?:[-–—]|\|+)\s*")


def _same(a: str, b: str) -> bool:
    return bool(a) and re.sub(r"\W", "", a).casefold() == re.sub(r"\W", "", b).casefold()


def search(query: str, n: int = 6) -> list[dict]:
    """YouTube search: [{url, title, artist, channel, duration, official}], official
    versions first. Official = the artist's own channel (or its Topic/VEVO channel)
    and not a cover, karaoke or lesson; that is how you tell the original apart."""
    query = query.strip()
    if not query:
        return []
    data = json.loads(_run(["-J", "--flat-playlist", f"ytsearch{n}:{query}"]))
    results = []
    for e in data.get("entries") or []:
        vid = e.get("id")
        if not vid or len(vid) != 11 or not e.get("duration"):  # channels, playlists, live streams
            continue
        raw_title, channel = e.get("title") or "", (e.get("channel") or e.get("uploader") or "").strip()
        own = re.sub(r"\s*(- Topic|VEVO)$", "", channel)
        # whose song: a dash-separated part of the title that is the channel, else the
        # first part ("Artist - Song" is the usual way), else the channel itself
        segs = [x.strip() for x in _DASH.split(_TAIL.sub("", raw_title)) if x.strip(" |")]
        mine = next((i for i, x in enumerate(segs) if _same(x, own)), None)
        if mine is not None and len(segs) > 1:
            artist, rest = own, segs[:mine] + segs[mine + 1 :]
        elif len(segs) > 1:
            artist, rest = segs[0], segs[1:]
        else:
            artist, rest = own, segs
        title, artist = clean_title(" - ".join(rest), artist)
        official = _same(artist, own) and not _NOT_ORIGINAL.search(raw_title)
        results.append({"url": f"https://www.youtube.com/watch?v={vid}", "title": title, "artist": artist,
                        "channel": channel, "duration": int(e["duration"]), "official": official})
    results.sort(key=lambda r: not r["official"])  # stable: YouTube's order otherwise
    return results
