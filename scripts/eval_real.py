"""Week-2 gate: transcription-first vs chroma on real songs with .lab ground truth.

uv run python scripts/eval_real.py [--systems chroma,muscriptor-medium,...] [--songs id1,id2]

Songs come from songs/manifest.csv; only those with include=y and a
songs/labels/<id>.lab are scored. Transcriptions are cached in cache/, so a
re-run after tuning harmony costs seconds."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from chordsheet.evaluation.real import METRICS, load_manifest, parse_system, score, sheet_intervals, write_lab
from chordsheet.pipeline import analyse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEMS = "chroma,muscriptor-small,muscriptor-medium,muscriptor-large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS)
    ap.add_argument("--songs", default="", help="comma-separated song ids (default: all labelled)")
    ap.add_argument("--self-transition", type=float, default=0.85)
    ap.add_argument("--beats", default="hybrid", help="beat tracker: hybrid | librosa | beat-this")
    ap.add_argument("--rich", action="store_true", help="7ths, sus4 and dim as well as major/minor")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    console = Console()
    systems = args.systems.split(",")
    wanted = set(filter(None, args.songs.split(",")))
    songs = [s for s in load_manifest() if s.include and (not wanted or s.id in wanted)]
    missing = [s.id for s in songs if not s.lab.exists()]
    songs = [s for s in songs if s.lab.exists()]
    if missing:
        console.print(f"[yellow]no ground truth yet, skipped:[/] {', '.join(missing)}")
    if not songs:
        raise SystemExit("nothing to score: add songs/labels/<id>.lab files")

    rows = []
    for song in songs:
        for system in systems:
            transcriber, size = parse_system(system)
            console.print(f"[dim]{song.id} · {system}[/]")
            sheet = analyse(
                song.url,
                transcriber=transcriber,
                model_size=size or "medium",
                self_transition=args.self_transition,
                rich=args.rich,
                beats=args.beats,
                cache_root=ROOT / "cache",
            )
            intervals, labels = sheet_intervals(sheet)
            write_lab(ROOT / "cache" / sheet["cache_key"] / f"est.{sheet['transcriber']}.lab", intervals, labels)
            rows.append({"song": song.id, "system": system, "key": sheet["key"], **score(song.lab, intervals, labels)})

    table = Table(title=f"Real set · {len(songs)} songs · self-transition {args.self_transition} · {'rich' if args.rich else 'major/minor'} vocabulary · {args.beats} beats")
    for col in ("song", "system", "key", *METRICS):
        table.add_column(col, justify="right" if col in METRICS else "left")
    for r in rows:
        table.add_row(r["song"], r["system"], r["key"], *(f"{r[m]:.3f}" for m in METRICS))
    for system in systems:
        mine = [r for r in rows if r["system"] == system]
        table.add_row("[bold]MEAN[/]", f"[bold]{system}[/]", "", *(f"[bold]{sum(r[m] for r in mine) / len(mine):.3f}[/]" for m in METRICS))
    console.print(table)

    args.out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(args.out / f"real-{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / f"real-{stamp}.json").write_text(json.dumps({"args": vars(args) | {"out": str(args.out)}, "rows": rows}, indent=1))
    console.print(f"[dim]saved results/real-{stamp}.csv[/]")


if __name__ == "__main__":
    main()
