"""Tune the close-call threshold (harmony.analyse.UNSURE_MARGIN) on the real songs.

For each chord run in each labelled song's sheet, is it right (mir_eval majmin over
most of its length) and how far did it lead its runner-up? The Stage 4 gate: at
least half of the wrong chords marked unsure, under 10% of the right ones.

    uv run python scripts/tune_confidence.py [--system muscriptor-large]
"""

from __future__ import annotations

import argparse

import mir_eval
import numpy as np

from chordsheet.evaluation.real import load_manifest, parse_system
from chordsheet.pipeline import analyse


def runs(sheet: dict) -> list[dict]:
    """Consecutive beats on the same chord -> {start, end, label, conf, alt}."""
    out: list[dict] = []
    for bar in sheet["bars"]:
        for b in bar:
            if out and out[-1]["label"] == b["mir_eval"] and abs(out[-1]["end"] - b["start"]) < 1e-3:
                out[-1]["end"] = b["end"]
            else:
                out.append({"start": b["start"], "end": b["end"], "label": b["mir_eval"],
                            "conf": b.get("conf"), "alt": b.get("alt"), "unsure": b.get("unsure")})
    return [r for r in out if r["label"] != "N" and r["conf"] is not None]


def right_fraction(run: dict, ref_iv: np.ndarray, ref_lab: list[str]) -> float | None:
    """Share of the run's scored time where the reference agrees (majmin); None if unscored."""
    good = total = 0.0
    for (s, e), lab in zip(ref_iv, ref_lab):
        o = min(e, run["end"]) - max(s, run["start"])
        if o <= 0 or lab == "X":
            continue
        cmp = mir_eval.chord.majmin([lab], [run["label"]])[0]
        if cmp < 0:  # reference outside the majmin vocabulary
            continue
        total += o
        good += o * cmp
    return good / total if total else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="muscriptor-large")
    args = ap.parse_args()
    name, size = parse_system(args.system)
    rows = []  # (conf, wrong, seconds)
    for song in load_manifest():
        if not song.include or not song.lab.exists():
            continue
        sheet = analyse(song.url, transcriber=name, model_size=size or "medium")
        ref_iv, ref_lab = mir_eval.io.load_labeled_intervals(str(song.lab))
        for r in runs(sheet):
            f = right_fraction(r, ref_iv, ref_lab)
            if f is not None:
                rows.append((r["conf"], f < 0.5, r["unsure"]))
        print(f"{song.id}: {len(runs(sheet))} chords")
    conf = np.array([r[0] for r in rows])
    wrong = np.array([r[1] for r in rows])
    print(f"\n{len(rows)} chords scored, {wrong.sum()} wrong ({wrong.mean():.0%})")
    print(f"median lead: right {np.median(conf[~wrong]):.3f}, wrong {np.median(conf[wrong]):.3f}\n")
    marked = np.array([r[2] for r in rows])
    print(f"current rule (analyse.UNSURE_MARGIN etc.): {(marked & wrong).sum() / max(wrong.sum(), 1):.0%} of wrong "
          f"chords marked, {(marked & ~wrong).sum() / max((~wrong).sum(), 1):.0%} of right ones\n")
    print("margin alone:")
    print("threshold  wrong marked  right marked  share of marks that are wrong")
    for t in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.2):
        m = conf < t
        caught, false = (m & wrong).sum() / max(wrong.sum(), 1), (m & ~wrong).sum() / max((~wrong).sum(), 1)
        prec = (m & wrong).sum() / max(m.sum(), 1)
        print(f"  {t:5.2f}     {caught:6.0%}        {false:6.0%}        {prec:6.0%}")


if __name__ == "__main__":
    main()
