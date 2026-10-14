# Real-song evaluation set

`manifest.csv` lists the songs (`include=n` keeps a song out of scoring);
`labels/<id>.lab` holds hand-checked ground truth, one chord per line:

```
0.000	2.412	N
2.412	4.830	G:maj
4.830	7.240	D:maj/3
```

Labels use mir_eval syntax (`C:maj`, `A:min7`, `G:7`, `F:maj7`, `D:sus4`, `B:dim`, `N`).
Slash bass is written as an interval (`D:maj/3` = D/F#) or omitted.

## Making ground truth

1. Paste the song's chart (Ultimate Guitar style: section headings, chords over
   lyrics, capo line) into `charts/<id>.txt`.
2. `uv run python scripts/draft_lab.py <id>` parses the chart, applies the capo, and
   force-aligns its chord sequence to the audio: the chart decides which chords,
   the audio only decides when they change. It also tries all 12 keys in case the
   chart is transposed, and lists low-fit stretches to listen to first.
3. Open the printed audio in [Sonic Visualiser](https://www.sonicvisualiser.org/),
   import `labels/<id>.draft.lab` (File → Import Annotation Layer), fix what is off,
   export as `labels/<id>.lab`, and delete the draft.

Charts often skip intros, instrumental breaks and extra repeats, and the aligner
can only stretch the chart's chords over those, so check them by ear. Charts are
git-ignored (third-party content); the corrected `.lab` files are committed.

Without a chart, `draft_lab.py` falls back to drafting from the system's own
output, which biases the ground truth toward that system. Avoid it for the eval set.

## Scoring

```
uv run python scripts/eval_real.py                                # all systems, all labelled songs
uv run python scripts/eval_real.py --systems chroma,muscriptor-medium --songs hated,runnin
```

Week-2 gate: MuScriptor beats chroma on major-minor accuracy (targets: 80% root, 70% majmin).
