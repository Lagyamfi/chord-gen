from __future__ import annotations

import subprocess
from pathlib import Path

from chordsheet.cache import DEFAULT_ROOT, Cache, sha256_file

SAMPLE_RATE = 44100


def decode_to_wav(src: Path, dst: Path, sr: int = SAMPLE_RATE) -> None:
    """Decode any ffmpeg-readable audio to mono WAV at `sr`."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(sr), str(dst)],
        check=True,
    )


def ingest_file(path: str | Path, cache_root: Path = DEFAULT_ROOT) -> tuple[Cache, Path]:
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    cache = Cache(sha256_file(src), cache_root)
    wav = cache.path("audio.wav")
    if not wav.exists():
        decode_to_wav(src, wav)
    return cache, wav
