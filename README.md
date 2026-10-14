# ChordSheet

Song in, beat-aligned guitar chord sheet out, with chord names **and** Roman numerals.
Transcribe first (MuScriptor → per-instrument MIDI), then reason about chords symbolically.

## Setup (macOS, Apple Silicon)

```bash
brew install ffmpeg fluid-synth yt-dlp
uv sync                                  # core + chroma fallback
uv sync --extra muscriptor --extra eval  # MuScriptor (torch, MPS) + mir_eval
```

MuScriptor weights are gated on Hugging Face: accept the conditions at
<https://huggingface.co/MuScriptor/muscriptor-large>, then `uv run hf auth login`.
Weights are **CC-BY-NC** — personal use only.

For the synthetic test set, put a General MIDI soundfont in `assets/soundfonts/`
(e.g. `FluidR3Mono_GM.sf3` from the MuseScore repo) or set `CHORDSHEET_SOUNDFONT`.
Falls back to the soundfont bundled with Homebrew's fluid-synth.

## Usage

```bash
uv run chordsheet analyse song.mp3                    # chroma fallback (fast, no model)
uv run chordsheet analyse song.mp3 -t muscriptor      # transcription-first (medium model)
uv run chordsheet analyse song.mp3 -t muscriptor --model-size large
uv run chordsheet analyse "https://youtu.be/..." -t muscriptor
uv run chordsheet analyse song.mp3 --rich --self-transition 0.9 --json   # --rich: 7ths, sus4, dim
```

Each layer is cached in `cache/<sha256>/` (`audio.wav`, `notes.<transcriber>.json` (e.g. `notes.muscriptor-medium.json`),
`transcription.<transcriber>.mid`, `beats.json`, `chords.json`, `sheet.json`), so iterating on
harmony never re-runs the model. Open `transcription.<transcriber>.mid` in MuseScore to see what
MuScriptor heard.

## Layout

```
src/chordsheet/
  ingest/       file, youtube, loopback (stub), chunk
  transcribe/   base (Transcriber protocol), muscriptor, chroma_fallback
  harmony/      beats, profile, templates, viterbi, key, analyse
  guitar/       bars, render            (capo + voicings: week 3)
  evaluation/   synthetic               (fixture generation + accuracy)
  pipeline.py   cache.py   cli.py
```

## Evaluation

```bash
uv run pytest
uv run python scripts/eval_synthetic.py                 # week-1 gate: >= 95% beat accuracy
uv run python scripts/eval_synthetic.py --transcriber muscriptor
```
