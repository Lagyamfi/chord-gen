"""Real-song evaluation: our chord output vs hand-checked .lab ground truth,
scored with the MIREX chord metrics from mir_eval (install the `eval` extra)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SONGS_DIR = Path(__file__).resolve().parents[3] / "songs"
METRICS = ("root", "majmin", "sevenths")


@dataclass(frozen=True)
class Song:
    id: str
    title: str
    artist: str
    url: str
    include: bool

    @property
    def lab(self) -> Path:
        return SONGS_DIR / "labels" / f"{self.id}.lab"


def load_manifest(path: Path = SONGS_DIR / "manifest.csv") -> list[Song]:
    with open(path, newline="") as f:
        return [
            Song(r["id"], r["title"], r["artist"], r["url"], r.get("include", "y").strip().lower() == "y")
            for r in csv.DictReader(f)
        ]


def parse_system(system: str) -> tuple[str, str]:
    """'chroma' -> ('chroma', ''), 'muscriptor-large' -> ('muscriptor', 'large')."""
    name, _, size = system.partition("-")
    if name == "muscriptor":
        return name, size or "medium"
    return name, ""


def sheet_intervals(sheet: dict) -> tuple[np.ndarray, list[str]]:
    """Beat-level chords from a sheet, merged into (intervals, mir_eval labels)."""
    intervals: list[list[float]] = []
    labels: list[str] = []
    for bar in sheet["bars"]:
        for b in bar:
            if labels and labels[-1] == b["mir_eval"] and abs(intervals[-1][1] - b["start"]) < 1e-3:
                intervals[-1][1] = b["end"]
            else:
                intervals.append([b["start"], b["end"]])
                labels.append(b["mir_eval"])
    return np.array(intervals).reshape(-1, 2), labels


def write_lab(path: Path, intervals: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{s:.3f}\t{e:.3f}\t{lab}\n" for (s, e), lab in zip(intervals, labels)))


def score(ref_lab: Path, est_intervals: np.ndarray, est_labels: list[str]) -> dict[str, float]:
    import mir_eval

    ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(str(ref_lab))
    scores = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
    return {m: float(scores[m]) for m in METRICS}
