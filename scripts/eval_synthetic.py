"""Week-1 gate: beat-level triad accuracy on the synthetic set.

uv run python scripts/eval_synthetic.py [--transcriber chroma|muscriptor]
"""

import argparse
import tempfile
from pathlib import Path

from chordsheet.evaluation.synthetic import PROGRESSIONS, beat_accuracy, make_fixture
from chordsheet.pipeline import analyse

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "generated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcriber", default="chroma")
    ap.add_argument("--model-size", default="medium", help="MuScriptor checkpoint")
    ap.add_argument("--self-transition", type=float, default=0.85)
    ap.add_argument("--gate", type=float, default=0.95)
    ap.add_argument("--beats", default="hybrid", help="beat tracker: hybrid | librosa | beat-this")
    args = ap.parse_args()

    scores = []
    with tempfile.TemporaryDirectory() as cache_root:
        for name in PROGRESSIONS:
            wav, segments = make_fixture(name, FIXTURES)
            sheet = analyse(
                str(wav),
                transcriber=args.transcriber,
                self_transition=args.self_transition,
                cache_root=Path(cache_root),
                model_size=args.model_size,
                beats=args.beats,
            )
            acc = beat_accuracy([b for bar in sheet["bars"] for b in bar], segments)
            scores.append(acc)
            print(f"{name:14s} key={sheet['key']:10s} acc={acc:.3f}")

    mean = sum(scores) / len(scores)
    verdict = "PASS" if mean >= args.gate else "FAIL"
    print(f"\nmean beat accuracy ({sheet['transcriber']}): {mean:.3f}  gate {args.gate:.2f}: {verdict}")
    raise SystemExit(0 if mean >= args.gate else 1)


if __name__ == "__main__":
    main()
