from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from chordsheet.cache import DEFAULT_ROOT, Cache, sha256_text
from chordsheet.ingest.file import decode_to_wav

_YT = re.compile(r"(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([\w-]{11})")


def is_youtube_url(source: str) -> bool:
    return bool(_YT.search(source))


def ingest_youtube(url: str, cache_root: Path = DEFAULT_ROOT) -> tuple[Cache, Path]:
    """Download audio with yt-dlp; the video id is the cache key."""
    match = _YT.search(url)
    if not match:
        raise ValueError(f"not a YouTube URL: {url}")
    cache = Cache(sha256_text(f"youtube:{match.group(1)}"), cache_root)
    wav = cache.path("audio.wav")
    if not wav.exists():
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dl.%(ext)s"
            subprocess.run(
                ["yt-dlp", "-q", "-x", "--audio-format", "wav", "-o", str(out), url],
                check=True,
            )
            decode_to_wav(Path(tmp) / "dl.wav", wav)
    return cache, wav
