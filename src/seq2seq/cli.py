from __future__ import annotations

import argparse
import json

from seq2seq.config import load_config
from seq2seq.data import prepare_data
from seq2seq.engine import evaluate, train, translate


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seq2seq", description="From-scratch RNN/LSTM encoder-decoder experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train_cmd = sub.add_parser("train", help="Train and save best/last checkpoints")
    train_cmd.add_argument("--config", "-c", default="configs/opus_books_lstm.yaml")
    train_cmd.add_argument("--output", default="runs/lstm-en-fr")
    train_cmd.add_argument("--resume", default=None)
    inspect_cmd = sub.add_parser("inspect-data", help="Show dataset statistics")
    inspect_cmd.add_argument("--config", "-c", default="configs/opus_books_lstm.yaml")
    eval_cmd = sub.add_parser("evaluate", help="Evaluate a trusted checkpoint")
    eval_cmd.add_argument("--checkpoint", required=True)
    eval_cmd.add_argument("--split", choices=["train", "validation", "test"], default="test")
    eval_cmd.add_argument("--max-examples", type=int, default=200)
    eval_cmd.add_argument("--device", default="auto")
    infer_cmd = sub.add_parser("translate", help="Generate translations")
    infer_cmd.add_argument("--checkpoint", required=True)
    infer_cmd.add_argument("--text", action="append", required=True)
    infer_cmd.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.command == "train":
        train(load_config(args.config), args.output, args.resume)
    elif args.command == "inspect-data":
        bundle = prepare_data(load_config(args.config))
        result = {
            "train": len(bundle.train), "validation": len(bundle.validation),
            "test": len(bundle.test), "src_vocab": len(bundle.src_vocab),
            "tgt_vocab": len(bundle.tgt_vocab), "fingerprint": bundle.fingerprint,
            "examples": bundle.train[:3],
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "evaluate":
        print(json.dumps(
            evaluate(args.checkpoint, args.split, args.max_examples, args.device),
            indent=2, ensure_ascii=False,
        ))
    elif args.command == "translate":
        for src, tgt in zip(args.text, translate(args.checkpoint, args.text, args.device)):
            print(json.dumps({"source": src, "prediction": tgt}, ensure_ascii=False))


if __name__ == "__main__":
    main()
