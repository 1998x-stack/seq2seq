from __future__ import annotations

import argparse
import json

from seq2seq.config import load_config
from seq2seq.data import prepare_data
from seq2seq.engine import evaluate, train, translate
from seq2seq.experiments import make_plan, run_matrix, summarize
from seq2seq.factory.experiments import factory_plan, factory_run
from seq2seq.telemetry import export_metrics_csv
from seq2seq.visualize import render_dashboard


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
    eval_cmd.add_argument("--predictions", default=None,
                          help="Optional JSONL path for generated predictions")
    infer_cmd = sub.add_parser("translate", help="Generate translations")
    infer_cmd.add_argument("--checkpoint", required=True)
    infer_cmd.add_argument("--text", action="append", required=True)
    infer_cmd.add_argument("--device", default="auto")
    experiment = sub.add_parser("experiments", help="Plan/run/summarize six ablations")
    experiment.add_argument("action", choices=["plan", "run", "summarize"])
    experiment.add_argument("--config", "-c", default="configs/opus_books_lstm.yaml")
    experiment.add_argument("--output", default="runs/stage02")
    experiment.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    experiment.add_argument("--split-seed", type=int, default=2026)
    experiment.add_argument("--max-examples", type=int, default=200)
    experiment.add_argument("--limit", type=int, default=None,
                            help="Run only the first N matrix entries (CI smoke)")
    experiment.add_argument("--dashboard", action="store_true",
                            help="Generate offline HTML dashboard after matrix run")
    dashboard = sub.add_parser("visualize", help="Build interactive offline HTML dashboard")
    dashboard.add_argument("--root", default="runs/stage02")
    dashboard.add_argument("--output", default=None)
    csv_cmd = sub.add_parser("export-csv", help="Export one run's epoch metrics to CSV")
    csv_cmd.add_argument("--run", required=True)
    paper = sub.add_parser("factory", help="Paper-grounded declarative factory experiments")
    paper.add_argument("action", choices=["plan", "run", "summarize", "visualize"])
    paper.add_argument("--spec", default="configs/factory/paper_ablation.yaml")
    paper.add_argument("--output", default="runs/paper-factory")
    paper.add_argument("--limit", type=int, default=None)
    paper.add_argument("--max-examples", type=int, default=200)
    paper.add_argument("--split", choices=["validation", "test", "train"],
                       default="validation")
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
            evaluate(args.checkpoint, args.split, args.max_examples, args.device, args.predictions),
            indent=2, ensure_ascii=False,
        ))
    elif args.command == "experiments":
        if args.action == "plan":
            plan = make_plan(load_config(args.config), args.seeds, args.split_seed)
            for item in plan:
                item.pop("config")
            print(json.dumps(plan, indent=2, ensure_ascii=False))
        elif args.action == "run":
            run_matrix(
                args.config, args.output, args.seeds, args.split_seed,
                args.max_examples, args.limit,
            )
            print(json.dumps(summarize(args.output), indent=2, ensure_ascii=False))
            if args.dashboard:
                print(render_dashboard(args.output))
        else:
            print(json.dumps(summarize(args.output), indent=2, ensure_ascii=False))
    elif args.command == "factory":
        if args.action == "plan":
            plan = factory_plan(args.spec)
            print(json.dumps([
                {k: v for k, v in entry.items() if k != "config"}
                for entry in plan
            ], indent=2, ensure_ascii=False))
        elif args.action == "run":
            report = factory_run(
                args.spec, args.output, max_examples=args.max_examples,
                limit=args.limit, split=args.split,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
        elif args.action == "summarize":
            print(json.dumps(summarize(args.output), indent=2, ensure_ascii=False))
        else:
            print(render_dashboard(args.output))
    elif args.command == "visualize":
        print(render_dashboard(args.root, args.output))
    elif args.command == "export-csv":
        print(export_metrics_csv(args.run))
    elif args.command == "translate":
        for src, tgt in zip(args.text, translate(args.checkpoint, args.text, args.device)):
            print(json.dumps({"source": src, "prediction": tgt}, ensure_ascii=False))


if __name__ == "__main__":
    main()
