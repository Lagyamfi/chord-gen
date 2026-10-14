"""Write a DRAFT ground-truth .lab for a song, to be checked by ear (songs/README.md).

With a pasted chart at songs/charts/<id>.txt (Ultimate Guitar style), the chart's
chord sequence is force-aligned to the audio: chords from the chart, timing from
the audio. Without one, the draft is the system's own output (biased toward it).

uv run python scripts/draft_lab.py <song-id> [--system muscriptor-large] [--transpose N]
"""

import argparse
from pathlib import Path

import numpy as np

from chordsheet.evaluation.align import align
from chordsheet.evaluation.chart import parse_chart
from chordsheet.evaluation.real import (
    SONGS_DIR,
    load_manifest,
    parse_system,
    sheet_intervals,
    write_lab,
)
from chordsheet.harmony.analyse import analyse_harmony
from chordsheet.harmony.profile import beat_profiles
from chordsheet.pipeline import analyse, load_layers

ROOT = Path(__file__).resolve().parents[1]
LOW_FIT = 0.6


def merge(windows, labels):
    intervals, merged = [], []
    for (s, e), lab in zip(windows, labels):
        if merged and merged[-1] == lab:
            intervals[-1][1] = e
        else:
            intervals.append([s, e])
            merged.append(lab)
    return np.array(intervals), merged


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def parse_time(value: str) -> float:
    """'55', '55.2' or '0:55' -> seconds."""
    if ":" in value:
        m, sec = value.split(":")
        return int(m) * 60 + float(sec)
    return float(value)


def from_chart(
    song,
    chart_path: Path,
    system: str,
    transpose: int | None,
    loops: bool,
    start: float,
    end: float,
) -> Path:
    chart = parse_chart(chart_path.read_text())
    seq = chart.sequence()
    print(
        f"chart: {len(chart.sections)} sections, {len(seq)} chord changes, capo {chart.capo}"
    )
    for sec in chart.sections:
        tag = " (repeated from earlier)" if sec.reused else ""
        rep = f" x{sec.repeat}" if sec.repeat > 1 else ""
        print(
            f"  [{sec.name}]{rep}{tag}: {' '.join(c.transpose(chart.capo).label for c in sec.chords)}"
        )

    transcriber, size = parse_system(system)
    cache, _, notes, grid = load_layers(
        song.url, transcriber, size or "medium", ROOT / "cache"
    )
    all_windows = grid.windows()
    anchors = chart.anchors()
    if anchors and anchors[0][0] == 0 and start == 0:
        start = anchors[0][
            1
        ]  # an anchored first section marks where the song proper starts
    free = analyse_harmony(notes, grid)
    # Only the song proper is aligned; talk, tuning and applause outside
    # [start, end] become X, which mir_eval leaves out of scoring.
    keep = [
        i
        for i, (s, e) in enumerate(all_windows)
        if s >= start - 1e-6 and e <= end + 1e-6
    ]
    if not keep:
        raise SystemExit(f"no beats between {fmt(start)} and {fmt(end)}")
    windows = [all_windows[i] for i in keep]
    heard = [free.chords[i].name(free.bass_pc[i]) for i in keep]
    profiles, bass_pc = beat_profiles(notes, windows)
    starts = np.array([w[0] for w in windows])
    beat_anchors = [(pos, int(np.abs(starts - t).argmin())) for pos, t, _ in anchors]
    for (_, t, name), (_, b) in zip(anchors, beat_anchors):
        print(f"anchor: [{name}] at {fmt(t)} -> beat {fmt(starts[b])}")
    result = align(
        seq,
        profiles,
        bass_pc,
        transpose,
        allow_loops=loops,
        times=list(starts),
        anchors=beat_anchors,
    )
    print(
        f"\naligning {fmt(windows[0][0])}-{fmt(windows[-1][1])}; outside that is X (not scored)"
    )

    best, runner_up = (
        result.candidates[0],
        result.candidates[1] if len(result.candidates) > 1 else None,
    )
    print(
        f"\nkey shift applied to chart: {result.transpose:+d} semitones (fit {best[1]:.3f}"
        + (
            f"; next best {runner_up[0]:+d} at {runner_up[1]:.3f})"
            if runner_up
            else ")"
        )
    )
    if result.transpose:
        print("  -> the recording is in a different key from the chart; check by ear")

    if result.loops:
        print("\nsections repeated beyond the chart (live repeats?):")
        for name, t in result.loops:
            print(f"  {fmt(t)}  [{name}] again")

    # Flag stretches where the assigned chord fits the audio poorly
    print(
        f"\nlow-fit stretches (fit < {LOW_FIT}), listen to these first; 'heard' is the {system} reading:"
    )
    flagged, t = 0, 0
    while t < len(windows):
        if result.fit[t] < LOW_FIT and result.labels[t] != "N":
            u = t
            while (
                u + 1 < len(windows)
                and result.fit[u + 1] < LOW_FIT
                and result.labels[u + 1] == result.labels[t]
            ):
                u += 1
            if u - t >= 1:  # at least two beats
                print(
                    f"  {fmt(windows[t][0])}-{fmt(windows[u][1])}  chart {result.labels[t]:10s} [{result.sections[t]}]"
                    f"  fit {result.fit[t : u + 1].mean():.2f}  heard {' '.join(dict.fromkeys(heard[t : u + 1]))}"
                )
                flagged += 1
            t = u + 1
        else:
            t += 1
    if not flagged:
        print("  none")

    out = SONGS_DIR / "labels" / f"{song.id}.draft.lab"
    intervals, labels = merge(windows, result.labels)
    song_end = all_windows[-1][1]
    if windows[0][0] > 0:
        intervals, labels = np.vstack([[0.0, windows[0][0]], intervals]), ["X", *labels]
    if windows[-1][1] < song_end:
        intervals, labels = (
            np.vstack([intervals, [windows[-1][1], song_end]]),
            [*labels, "X"],
        )
    write_lab(out, intervals, labels)
    print(f"\naudio: {cache.path('audio.wav')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song_id")
    ap.add_argument(
        "--system", default="muscriptor-large", help="notes used for timing"
    )
    ap.add_argument(
        "--chart", type=Path, help="default: songs/charts/<id>.txt if it exists"
    )
    ap.add_argument(
        "--transpose", type=int, help="force the chart key shift instead of searching"
    )
    ap.add_argument(
        "--loops",
        action="store_true",
        help="let sections repeat beyond the chart (live repeats)",
    )
    ap.add_argument(
        "--start",
        default="0",
        help="where the song proper starts, e.g. 0:55 (earlier is X)",
    )
    ap.add_argument(
        "--end", default="", help="where it ends (later is X); default: end of audio"
    )
    args = ap.parse_args()

    song = next((s for s in load_manifest() if s.id == args.song_id), None)
    if song is None:
        raise SystemExit(f"unknown song id: {args.song_id}")
    chart = args.chart or SONGS_DIR / "charts" / f"{song.id}.txt"

    if chart.exists():
        out = from_chart(
            song,
            chart,
            args.system,
            args.transpose,
            args.loops,
            parse_time(args.start),
            parse_time(args.end) if args.end else float("inf"),
        )
    else:
        print(
            f"no chart at {chart}; drafting from {args.system} output (biased toward it)"
        )
        transcriber, size = parse_system(args.system)
        sheet = analyse(
            song.url,
            transcriber=transcriber,
            model_size=size or "medium",
            cache_root=ROOT / "cache",
        )
        out = SONGS_DIR / "labels" / f"{song.id}.draft.lab"
        write_lab(out, *sheet_intervals(sheet))
    print(f"draft: {out}\nCheck it by ear, then rename to {song.id}.lab")


if __name__ == "__main__":
    main()
