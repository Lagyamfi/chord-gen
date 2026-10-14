from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import typer

from chordsheet.cache import DEFAULT_ROOT

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Song in, guitar chord sheet out.")


class TranscriberName(str, Enum):
    chroma = "chroma"
    muscriptor = "muscriptor"


@app.command()
def analyse(
    source: str = typer.Argument(..., help="Audio file path or YouTube URL"),
    transcriber: TranscriberName = typer.Option(TranscriberName.chroma, "--transcriber", "-t"),
    self_transition: float = typer.Option(0.85, help="Viterbi stay probability; higher = fewer chord changes"),
    no_sevenths: bool = typer.Option(False, "--no-sevenths", help="Collapse to triads"),
    cache_dir: Path = typer.Option(DEFAULT_ROOT, help="Cache root"),
    refresh: bool = typer.Option(False, help="Recompute cached layers"),
    as_json: bool = typer.Option(False, "--json", help="Print sheet JSON instead of the chart"),
):
    """Analyse a song and print its chord chart."""
    from chordsheet.guitar.render import render_chart
    from chordsheet.pipeline import analyse as run

    sheet = run(
        source,
        transcriber=transcriber.value,
        self_transition=self_transition,
        sevenths=not no_sevenths,
        cache_root=cache_dir,
        refresh=refresh,
    )
    typer.echo(json.dumps(sheet, indent=1) if as_json else render_chart(sheet))
    typer.echo(f"\n(sheet: {cache_dir / sheet['cache_key'] / 'sheet.json'})", err=True)


@app.command()
def listen():
    """Live loopback mode (week 4)."""
    raise typer.Exit("loopback mode lands in week 4")
