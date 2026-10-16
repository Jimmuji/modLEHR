"""Command line: one command per stage, plus helpers to run and compare experiments.

    modlehr convert  synthea --raw-dir RAW --out data/synthea_meds
    modlehr validate --meds-dir data/synthea_meds
    modlehr prepare  --config configs/experiment/synthea_gpt.yaml
    modlehr info     --config configs/experiment/synthea_gpt.yaml
    modlehr train    --config configs/experiment/synthea_gpt.yaml
    modlehr evaluate --config configs/experiment/synthea_gpt.yaml
    modlehr baseline --config configs/experiment/synthea_gpt.yaml
    modlehr generate --config configs/experiment/synthea_gpt.yaml --subject 0 --start-age 50
    modlehr run      --config configs/experiment/synthea_gpt.yaml --seed 2
    modlehr compare  runs/synthea_gpt_s1 runs/synthea_gpt_rope_age_s1 runs/synthea_gpt_logistic

Every command that takes --config also takes --set section.key=value (repeatable) to change a
value without a new file, and --seed N, which sets train.seed and adds _sN to train.out_dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CONFIG_COMMANDS = {
    "prepare": "MEDS → tokenized timelines with labels",
    "info": "summarize a prepared dataset",
    "train": "train a model",
    "evaluate": "AUC at landmark ages and held-out likelihood",
    "baseline": "logistic regression and gradient boosting on the same task",
    "generate": "sample futures for one patient",
    "run": "prepare (if the data config changed), train and evaluate",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="modlehr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="convert a data source to MEDS")
    conv.add_argument("source", choices=["synthea"])
    conv.add_argument("--raw-dir", required=True, help="Synthea CSV export, one directory or chunk directories")
    conv.add_argument("--out", required=True, help="MEDS output directory")

    val = sub.add_parser("validate", help="check a MEDS dataset against the meds schemas")
    val.add_argument("--meds-dir", required=True)

    cmp_ = sub.add_parser("compare", help="compare evaluated runs against the first one")
    cmp_.add_argument("runs", nargs="+", help="run directories containing evaluate.csv")

    for name, text in CONFIG_COMMANDS.items():
        p = sub.add_parser(name, help=text)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                       help="override a config value, e.g. --set model.attention=gqa (repeatable)")
        p.add_argument("--seed", type=int, help="set train.seed and add _s<seed> to train.out_dir")
        if name in ("evaluate", "generate"):
            p.add_argument("--checkpoint", help="default: <train.out_dir>/best.pt")
        if name == "evaluate":
            p.add_argument("--out", help="default: <train.out_dir>/evaluate.csv")
        if name == "baseline":
            p.add_argument("--models", default="logistic,gbm", help="comma-separated: logistic, gbm")
            p.add_argument("--gbm-features", type=int, default=2000,
                           help="gradient boosting uses this many of the most common feature columns")
        if name in ("train", "run"):
            p.add_argument("--resume", action="store_true", help="continue from <train.out_dir>/last.pt")
            p.add_argument("--stop-after", type=int, metavar="STEP",
                           help="stop after this step, e.g. to fit a job's time limit; continue later with --resume")
        if name == "run":
            p.add_argument("--force-prepare", action="store_true", help="prepare even if the prepared data is current")
        if name == "generate":
            p.add_argument("--split", default="held_out")
            p.add_argument("--subject", type=int, default=0, help="index of the subject within the split")
            p.add_argument("--start-age", type=float, required=True)
            p.add_argument("--horizon", type=float, default=20.0)
            p.add_argument("--samples", type=int, default=3)
    return ap


def load(args: argparse.Namespace):
    from modlehr.config import apply_overrides, load_config

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.seed is not None:
        cfg.train.seed = args.seed
        cfg.train.out_dir = f"{cfg.train.out_dir}_s{args.seed}"
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up when output goes to a file

    if args.command == "convert":
        from modlehr.data.adapters.synthea import convert
        print(convert(args.raw_dir, args.out))
        return
    if args.command == "validate":
        from modlehr.data.meds_io import open_meds
        ds = open_meds(args.meds_dir, validate=True)
        print(f"{args.meds_dir}: {len(ds.shards())} shards, {len(ds.codes):,} codes, {len(ds.splits):,} subjects; valid")
        return
    if args.command == "compare":
        from modlehr.evaluate.compare import compare, print_comparison
        print_comparison(compare(args.runs))
        return

    cfg = load(args)
    checkpoint = getattr(args, "checkpoint", None) or str(Path(cfg.train.out_dir) / "best.pt")

    if args.command == "prepare":
        from modlehr.prepare.pipeline import run_prepare
        run_prepare(cfg.data)
    elif args.command == "info":
        from modlehr.prepare.pipeline import describe
        for line in describe(cfg.data):
            print(line)
    elif args.command == "train":
        from modlehr.train.loop import train
        train(cfg, resume=args.resume, stop_after=args.stop_after)
    elif args.command == "evaluate":
        from modlehr.evaluate.run import evaluate
        evaluate(cfg, checkpoint, args.out)
    elif args.command == "baseline":
        from modlehr.evaluate.baselines import run_baselines
        run_baselines(cfg, tuple(m.strip() for m in args.models.split(",") if m.strip()), args.gbm_features)
    elif args.command == "run":
        from modlehr.evaluate.run import evaluate
        from modlehr.prepare.pipeline import is_current, run_prepare
        from modlehr.train.loop import train
        if args.force_prepare or not is_current(cfg.data):
            run_prepare(cfg.data)
        else:
            print(f"[run] {cfg.data.prepared_dir} matches the data config; skipping prepare")
        best = train(cfg, resume=args.resume, stop_after=args.stop_after)
        if args.stop_after is None or args.stop_after >= cfg.train.steps - 1:
            evaluate(cfg, best)
    elif args.command == "generate":
        from modlehr.evaluate.run import load_model
        from modlehr.generate.sample import sample_future
        from modlehr.prepare.dataset import open_split
        from modlehr.train.loop import pick_device
        device = pick_device(cfg.train.device)
        model, vocab = load_model(checkpoint, Path(cfg.data.prepared_dir), device)
        split = open_split(cfg.data.prepared_dir, args.split)
        from modlehr.data.meds_io import open_meds
        names = {}
        if Path(cfg.data.meds_dir).exists():
            codes = open_meds(cfg.data.meds_dir).codes
            names = dict(zip(codes["code"].astype(str), codes["description"].fillna("").astype(str)))
        history = split.subject(args.subject)
        print(f"subject {int(split.subject_ids[args.subject])} ({args.split}), sampling from age {args.start_age:g} "
              f"for {args.horizon:g} years")
        for s in range(args.samples):
            events = sample_future(model, vocab, history, args.start_age, args.horizon, seed=s, device=device)
            print(f"sample {s}:" + ("" if events else " no outcome within the horizon"))
            for age, token in events:
                print(f"  {age:6.2f}  {token:30} {names.get(token, '')}")


if __name__ == "__main__":
    main()
