"""Minimum web player: serves cached audio and chord sheets to one static page."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chordsheet.guitar.progressions import label as progression_label
from chordsheet.guitar.progressions import library, section_progressions
from chordsheet.guitar.sections import PRESETS, sections_for

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


class Rename(BaseModel):
    times: list[float]  # start times (s) of the section runs to rename
    name: str


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
        # no-cache: the browser revalidates, so a new player shows on a plain refresh
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    def load_sheet(d: Path, system: str) -> dict:
        sheet = json.loads((d / f"sheet.{system}.json").read_text())
        sheet["sections"] = sections_for(sheet, load_names(d))  # renames apply to every system
        return sheet

    def library_songs() -> list[dict]:
        """Every analysed song with its best sheet: the progression library's input."""
        titles = _titles(manifest)
        out = []
        for d in sorted(cache_root.iterdir()) if cache_root.exists() else []:
            available = systems(d) if d.is_dir() else []
            if not available or not (d / "audio.wav").exists():
                continue
            sheet = load_sheet(d, available[0])
            meta = titles.get(sheet["source"], {})
            out.append({"id": d.name, "title": meta.get("title") or Path(sheet["source"]).stem,
                        "artist": meta.get("artist", ""), "systems": available, "sheet": sheet})
        return out

    @app.get("/api/songs")
    def songs():
        out = [
            {k: s[k] for k in ("id", "title", "artist", "systems")} | {"key": s["sheet"]["key"]}
            for s in library_songs()
        ]
        return sorted(out, key=lambda s: (s["artist"], s["title"]))

    @app.get("/api/songs/{key}/sheet")
    def sheet(key: str, system: str | None = None):
        d = song_dir(key)
        available = systems(d)
        chosen = system or (available[0] if available else None)
        if chosen not in available:
            raise HTTPException(404, f"no sheet for system {system!r}")
        sheet = load_sheet(d, chosen)
        sheet["section_presets"] = list(PRESETS)
        sheet["progressions"] = progression_tags(key, sheet)
        return sheet

    def progression_tags(key: str, sheet: dict, per_section: int = 2) -> list[list[dict]]:
        """Per section block, up to two progression tags, named or shared ones
        first, each with the places in OTHER songs that use the same loop."""
        lib = library(library_songs())
        out = []
        for items in section_progressions(sheet):
            tags = []
            for p in items:
                others = [o for o in lib.get(p["key"], []) if o["song"] != key]
                seen, uniq = set(), []
                for o in others:  # one link per song: its first section with this loop
                    if o["song"] not in seen:
                        seen.add(o["song"])
                        uniq.append({k: o[k] for k in ("song", "title", "section", "time")})
                tags.append({"label": progression_label(p), "name": p["name"], "line": p["line"],
                             "others": uniq, "songs": len({o["song"] for o in uniq})})
            tags.sort(key=lambda t: (-(t["songs"] > 0), -(t["name"] is not None)))  # stable: line order otherwise
            out.append(tags[:per_section])
        return out

    def load_names(d: Path) -> dict:
        f = d / "names.json"
        return json.loads(f.read_text()) if f.exists() else {}

    @app.post("/api/songs/{key}/names")
    def rename(key: str, body: Rename):
        name = body.name.strip()
        if not name or len(name) > 40:
            raise HTTPException(422, "name must be 1-40 characters")
        d = song_dir(key)
        names = load_names(d)
        names.update({f"{t:.1f}": name for t in body.times})
        (d / "names.json").write_text(json.dumps(names, indent=1))
        return {"saved": len(body.times)}

    @app.get("/api/songs/{key}/audio")
    def audio(key: str):
        # FileResponse honours Range requests, so the browser can seek
        return FileResponse(song_dir(key) / "audio.wav", media_type="audio/wav")

    return app
