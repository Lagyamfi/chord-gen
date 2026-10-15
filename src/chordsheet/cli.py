from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import typer

from chordsheet.cache import DEFAULT_ROOT

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Song in, guitar chord sheet out.")


def _check_key(value: str | None) -> str | None:
    if value is None:
        return None
    from chordsheet.harmony.key import parse_key

    try:
        parse_key(value)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    return value


KEY_HELP = "Set the key instead of detecting it, e.g. Bb, 'G minor', Em (changes Roman numerals only)"


class TranscriberName(str, Enum):
    chroma = "chroma"
    muscriptor = "muscriptor"


@app.command()
def analyse(
    source: str = typer.Argument(..., help="Audio file path or YouTube URL"),
    transcriber: TranscriberName = typer.Option(TranscriberName.chroma, "--transcriber", "-t"),
    model_size: str = typer.Option("medium", help="MuScriptor checkpoint: small | medium | large"),
    self_transition: float = typer.Option(0.85, help="Viterbi stay probability; higher = fewer chord changes"),
    capo: int = typer.Option(None, help="Force a capo fret (default: suggest one)"),
    key: str = typer.Option(None, help=KEY_HELP, callback=_check_key),
    beats: str = typer.Option("hybrid", help="hybrid (chords on librosa beats, bars from beat_this) | librosa | beat-this"),
    rich: bool = typer.Option(False, "--rich", help="Also detect 7ths, sus4 and dim (default: major/minor)"),
    cache_dir: Path = typer.Option(DEFAULT_ROOT, help="Cache root"),
    refresh: bool = typer.Option(False, help="Recompute cached layers"),
    as_json: bool = typer.Option(False, "--json", help="Print sheet JSON instead of the chart"),
    all_bars: bool = typer.Option(False, "--bars", help="Print every bar instead of the one-page song chart"),
):
    """Analyse a song and print its chord chart."""
    from chordsheet.guitar.render import render_chart, render_song
    from chordsheet.pipeline import analyse as run

    sheet = run(
        source,
        transcriber=transcriber.value,
        self_transition=self_transition,
        rich=rich,
        beats=beats,
        capo=capo,
        key=key,
        cache_root=cache_dir,
        refresh=refresh,
        model_size=model_size,
    )
    typer.echo(json.dumps(sheet, indent=1) if as_json else (render_chart if all_bars else render_song)(sheet))
    typer.echo(f"\n(sheet: {cache_dir / sheet['cache_key'] / 'sheet.json'})", err=True)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="0.0.0.0 to open it from your phone on the same Wi-Fi"),
    port: int = typer.Option(8642),
    cache_dir: Path = typer.Option(DEFAULT_ROOT, help="Cache root"),
    manifest: Path = typer.Option(Path("songs/manifest.csv"), help="Song titles"),
    add_model: str = typer.Option("large", help="MuScriptor size for songs added from a link in the player"),
):
    """Web player: play an analysed song with its chord chart."""
    import uvicorn

    from chordsheet.web.jobs import Jobs, analyse_link
    from chordsheet.web.server import create_app

    try:
        import muscriptor  # noqa: F401

        runner = analyse_link("muscriptor", add_model)
    except ImportError:
        typer.echo("MuScriptor not installed: songs added in the player use the chroma fallback.", err=True)
        runner = analyse_link("chroma")
    typer.echo(f"ChordSheet player on http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}")
    uvicorn.run(create_app(cache_dir, manifest, Jobs(cache_dir, runner)), host=host, port=port, log_level="warning")


@app.command()
def devices():
    """List audio input devices (to find your loopback device)."""
    from chordsheet.ingest.loopback import list_input_devices

    for d in list_input_devices():
        typer.echo(f"{d['index']:3d}  {d['name']}  ({d['channels']} ch, {d['samplerate']:.0f} Hz)")


@app.command()
def listen(
    transcriber: TranscriberName = typer.Option(TranscriberName.muscriptor, "--transcriber", "-t"),
    model_size: str = typer.Option("medium", help="MuScriptor checkpoint: small is fastest"),
    chunk: float = typer.Option(30.0, help="Seconds of audio per update"),
    device: str = typer.Option("BlackHole", help="Input device name (or part of it); see `chordsheet devices`"),
    title: str = typer.Option(None, help="Name for this session in the player"),
    capo: int = typer.Option(None, help="Force a capo fret"),
    key: str = typer.Option(None, help=KEY_HELP, callback=_check_key),
    cache_dir: Path = typer.Option(DEFAULT_ROOT, help="Cache root"),
):
    """Live mode: chords for whatever is playing on this computer, ~40 s behind."""
    import time

    from chordsheet.guitar.render import render_chart
    from chordsheet.ingest.loopback import LoopbackRecorder, find_input_device
    from chordsheet.live import LiveSession

    dev = find_input_device(device)
    if dev is None:
        typer.echo(
            f"No input device matching {device!r}. One-time macOS setup:\n"
            "  1. brew install blackhole-2ch  (then log out and in)\n"
            "  2. Audio MIDI Setup -> + -> Create Multi-Output Device: tick your speakers and BlackHole 2ch\n"
            "  3. System Settings -> Sound -> Output: the Multi-Output Device\n"
            "Then run `chordsheet devices` to check it shows up.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Loading {transcriber.value}{'-' + model_size if transcriber.value == 'muscriptor' else ''}...")
    session = LiveSession(int(dev["samplerate"]), transcriber.value, model_size, chunk, cache_root=cache_dir, title=title, capo=capo, key=key)
    shown_until = 0.0

    def show(sheet: dict) -> None:
        nonlocal shown_until
        new_bars = [bar for bar in sheet["bars"] if bar[-1]["end"] > shown_until + 0.05]
        text = render_chart({**sheet, "bars": new_bars})
        typer.echo("\n" + (text if shown_until == 0 else text.split("\n", 2)[2]))
        if session.last_chunk_silent:
            typer.echo("(that chunk was silent: is the Multi-Output Device your sound output?)", err=True)
        shown_until = session.done / session.sr

    typer.echo(f"Listening on {dev['name']}. First chords in ~{chunk:.0f}s plus model time. Ctrl-C to stop.")
    try:
        with LoopbackRecorder(dev) as rec:
            while True:
                time.sleep(0.25)
                sheet = session.feed(rec.read())
                if sheet:
                    show(sheet)
                else:
                    typer.echo(f"\r  recording {session.seconds:6.1f}s · next update in "
                               f"{max(0.0, chunk - session.pending_seconds):4.1f}s ", nl=False, err=True)
    except KeyboardInterrupt:
        typer.echo("\nStopping; transcribing the last bit...", err=True)
        sheet = session.finish()
        if sheet:
            show(sheet)
    typer.echo(f"\nSaved as {session.source!r}: open it with `chordsheet serve`.")
