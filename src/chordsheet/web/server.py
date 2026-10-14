"""Minimum web player: serves cached audio and chord sheets to one static page."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

STATIC = Path(__file__).parent / "static"
# best first: the player shows the most accurate sheet a song has
SYSTEM_PREFERENCE = ["muscriptor-large", "muscriptor-medium", "muscriptor-small", "chroma"]
_KEY = re.compile(r"^[0-9a-f]{64}$")


def _rank(system: str) -> int:
    return SYSTEM_PREFERENCE.index(system) if system in SYSTEM_PREFERENCE else len(SYSTEM_PREFERENCE)


def _titles(manifest: Path | None) -> dict[str, dict]:
    if not manifest or not manifest.exists():
        return {}
    with open(manifest, newline="") as f:
        return {r["url"]: {"title": r["title"], "artist": r["artist"]} for r in csv.DictReader(f)}


def create_app(cache_root: Path, manifest: Path | None = None) -> FastAPI:
    cache_root = Path(cache_root)
    app = FastAPI(title="ChordSheet player")

    def song_dir(key: str) -> Path:
        d = cache_root / key
        if not _KEY.match(key) or not (d / "audio.wav").exists():
            raise HTTPException(404, "unknown song")
        return d

    def systems(d: Path) -> list[str]:
        found = [p.name[len("sheet.") : -len(".json")] for p in d.glob("sheet.*.json")]
        return sorted(found, key=_rank)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/songs")
    def songs():
        titles = _titles(manifest)
        out = []
        for d in sorted(cache_root.iterdir()) if cache_root.exists() else []:
            available = systems(d) if d.is_dir() else []
            if not available or not (d / "audio.wav").exists():
                continue
            sheet = json.loads((d / f"sheet.{available[0]}.json").read_text())
            meta = titles.get(sheet["source"], {})
            out.append(
                {
                    "id": d.name,
                    "title": meta.get("title") or Path(sheet["source"]).stem,
                    "artist": meta.get("artist", ""),
                    "key": sheet["key"],
                    "systems": available,
                }
            )
        return sorted(out, key=lambda s: (s["artist"], s["title"]))

    @app.get("/api/songs/{key}/sheet")
    def sheet(key: str, system: str | None = None):
        d = song_dir(key)
        available = systems(d)
        chosen = system or (available[0] if available else None)
        if chosen not in available:
            raise HTTPException(404, f"no sheet for system {system!r}")
        return json.loads((d / f"sheet.{chosen}.json").read_text())

    @app.get("/api/songs/{key}/audio")
    def audio(key: str):
        # FileResponse honours Range requests, so the browser can seek
        return FileResponse(song_dir(key) / "audio.wav", media_type="audio/wav")

    return app
