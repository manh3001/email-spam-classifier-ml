"""Command-line interface for training and prediction."""

import argparse
import logging
import sys
from pathlib import Path

from .config import DATA_PATH, MODEL_PATH
from .predict import SpamClassifier
from .train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spam-classify")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train and persist the best model")
    t.add_argument("--data", default=None)
    t.add_argument("--model", default=str(MODEL_PATH))
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("predict", help="Classify a message or a file of messages")
    p.add_argument("message", nargs="?", default=None)
    p.add_argument("--file", default=None)
    p.add_argument("--model", default=str(MODEL_PATH))
    return parser


def _run_train(args) -> int:
    train(
        data_path=args.data or DATA_PATH,
        model_path=args.model,
        test_size=args.test_size,
        random_state=args.seed,
    )
    return 0


def _run_predict(args) -> int:
    try:
        clf = SpamClassifier.load(args.model)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.file:
        lines = [
            ln.strip()
            for ln in Path(args.file).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    elif args.message is not None:
        lines = [args.message]
    else:
        print("Error: provide a message or --file PATH", file=sys.stderr)
        return 2

    for text, result in zip(lines, clf.predict_batch(lines)):
        print(f"[{result['label']}] ({result['spam_probability']:.2f}) {text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    if args.command == "predict":
        return _run_predict(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
