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
    model_size: str = typer.Option("medium", help="MuScriptor checkpoint: small | medium | large"),
    self_transition: float = typer.Option(0.85, help="Viterbi stay probability; higher = fewer chord changes"),
    beats: str = typer.Option("hybrid", help="hybrid (chords on librosa beats, bars from beat_this) | librosa | beat-this"),
    rich: bool = typer.Option(False, "--rich", help="Also detect 7ths, sus4 and dim (default: major/minor)"),
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
        rich=rich,
        beats=beats,
        cache_root=cache_dir,
        refresh=refresh,
        model_size=model_size,
    )
    typer.echo(json.dumps(sheet, indent=1) if as_json else render_chart(sheet))
    typer.echo(f"\n(sheet: {cache_dir / sheet['cache_key'] / 'sheet.json'})", err=True)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="0.0.0.0 to open it from your phone on the same Wi-Fi"),
    port: int = typer.Option(8642),
    cache_dir: Path = typer.Option(DEFAULT_ROOT, help="Cache root"),
    manifest: Path = typer.Option(Path("songs/manifest.csv"), help="Song titles"),
):
    """Web player: play an analysed song with its chord chart."""
    import uvicorn

    from chordsheet.web.server import create_app

    typer.echo(f"ChordSheet player on http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}")
    uvicorn.run(create_app(cache_dir, manifest), host=host, port=port, log_level="warning")


@app.command()
def listen():
    """Live loopback mode (week 4)."""
    raise typer.Exit("loopback mode lands in week 4")
