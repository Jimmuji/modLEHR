"""Command line: one command per stage.

    modlehr convert synthea --raw-dir RAW --out data/synthea_meds
    modlehr validate --meds-dir data/synthea_meds
    modlehr prepare  --config configs/experiment/synthea_gpt.yaml
    modlehr train    --config configs/experiment/synthea_gpt.yaml
    modlehr evaluate --config configs/experiment/synthea_gpt.yaml
    modlehr generate --config configs/experiment/synthea_gpt.yaml --subject 0 --start-age 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="modlehr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="convert a data source to MEDS")
    conv.add_argument("source", choices=["synthea"])
    conv.add_argument("--raw-dir", required=True)
    conv.add_argument("--out", required=True)

    val = sub.add_parser("validate", help="check a MEDS dataset against the meds schemas")
    val.add_argument("--meds-dir", required=True)

    for name, text in (("prepare", "MEDS → tokenized timelines"), ("train", "train a model"),
                       ("evaluate", "AUC at landmark ages"), ("generate", "sample futures for one patient")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--config", required=True)
        if name in ("evaluate", "generate"):
            p.add_argument("--checkpoint", help="default: <train.out_dir>/best.pt")
        if name == "evaluate":
            p.add_argument("--out", help="default: <train.out_dir>/evaluate.csv")
        if name == "generate":
            p.add_argument("--split", default="held_out")
            p.add_argument("--subject", type=int, default=0, help="index of the subject within the split")
            p.add_argument("--start-age", type=float, required=True)
            p.add_argument("--horizon", type=float, default=20.0)
            p.add_argument("--samples", type=int, default=3)

    args = ap.parse_args(argv)
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

    from modlehr.config import load_config
    cfg = load_config(args.config)
    checkpoint = getattr(args, "checkpoint", None) or str(Path(cfg.train.out_dir) / "best.pt")

    if args.command == "prepare":
        from modlehr.prepare.pipeline import run_prepare
        run_prepare(cfg.data)
    elif args.command == "train":
        from modlehr.train.loop import train
        train(cfg)
    elif args.command == "evaluate":
        from modlehr.evaluate.run import evaluate
        evaluate(cfg, checkpoint, args.out)
    elif args.command == "generate":
        from modlehr.evaluate.run import load_model
        from modlehr.generate.sample import sample_future
        from modlehr.prepare.dataset import PreparedSplit
        from modlehr.train.loop import pick_device
        device = pick_device(cfg.train.device)
        model, vocab = load_model(checkpoint, Path(cfg.data.prepared_dir), device)
        split = PreparedSplit(Path(cfg.data.prepared_dir) / f"{args.split}.npz")
        history = split.subject(args.subject)
        for s in range(args.samples):
            events = sample_future(model, vocab, history, args.start_age, args.horizon, seed=s, device=device)
            print(f"sample {s}: " + (", ".join(f"{a}: {t}" for a, t in events) or "no events"))


if __name__ == "__main__":
    main()
