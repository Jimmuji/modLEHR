"""Command line: one command per stage, plus helpers to run and compare experiments.

    modlehr convert  toy --out data/toy_meds          (a synthetic dataset to try everything in a minute)
    modlehr convert  synthea --raw-dir RAW --out data/synthea_meds
    modlehr convert  mimic-iv-core --raw-dir RAW --out data/mimic_iv_core_meds   (diagnoses only)
    modlehr validate --meds-dir data/synthea_meds
    modlehr prepare  --config configs/experiment/synthea_gpt.yaml
    modlehr info     --config configs/experiment/synthea_gpt.yaml
    modlehr train    --config configs/experiment/synthea_gpt.yaml
    modlehr evaluate --config configs/experiment/synthea_gpt.yaml
    modlehr baseline --config configs/experiment/synthea_gpt.yaml
    modlehr generate --config configs/experiment/synthea_gpt.yaml --subject 0 --start-age 50
    modlehr simulate --config configs/experiment/synthea_gpt.yaml --start-age 50
    modlehr delphi   --config configs/experiment/mimic_iv_core_gpt.yaml --seed 1   (Delphi-2M's own AUC)
    modlehr export-delphi --config configs/experiment/toy_gpt.yaml   (the prepared data in Delphi-2M's format)
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
    "simulate": "sample one future per held-out patient and compare incidence with the records",
    "delphi": "Delphi-2M's own AUC (by sex and 5-year age band, against never-havers), to compare with its numbers",
    "export-delphi": "write the prepared data in Delphi-2M's format, with a config for its train.py",
    "run": "prepare (if the data config changed), train and evaluate",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="modlehr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="convert a data source to MEDS")
    conv.add_argument("source", choices=["synthea", "mimic-iv-core", "mimic-iv-note", "toy"])
    conv.add_argument("--raw-dir",
                      help="synthea: the CSV export; mimic-iv-core: the MIMIC-IV release directory with hosp/; "
                           "mimic-iv-note: the folder with discharge.csv.gz and radiology.csv.gz")
    conv.add_argument("--out", required=True, help="MEDS output directory")
    conv.add_argument("--meds-dir", help="mimic-iv-note: the MIMIC-IV MEDS dataset to add the notes to")
    conv.add_argument("--note-kinds", default="discharge,radiology", help="mimic-iv-note: which note files to add")
    conv.add_argument("--subjects", type=int, default=2000, help="toy: how many synthetic subjects")
    conv.add_argument("--diagnosis-time", choices=["discharge", "admission"], default="discharge",
                      help="mimic-iv-core: time a stay's diagnoses at its discharge (as the official ETL) or admission")
    conv.add_argument("--extra", default="", help="mimic-iv-core: inputs to add, comma-separated: procedures, medications")

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
        if name in ("evaluate", "generate", "simulate", "delphi"):
            p.add_argument("--checkpoint", help="default: <train.out_dir>/best.pt")
        if name == "export-delphi":
            p.add_argument("--out", help="default: <train.out_dir>_delphi_data")
            p.add_argument("--delphi-out", help="where Delphi's train.py writes its checkpoint; default: <train.out_dir>_delphi_code")
        if name == "simulate":
            p.add_argument("--start-age", type=float, required=True)
            p.add_argument("--subjects", type=int, default=1000, help="held-out patients to sample, at most")
            p.add_argument("--out-dir", help="default: <train.out_dir>")
        if name == "evaluate":
            p.add_argument("--out", help="default: <train.out_dir>/evaluate.csv")
        if name == "delphi":
            p.add_argument("--gap-days", default="0.1,365.25",
                           help="comma-separated gaps before each diagnosis; Delphi-2M reports 0.1 day and one year")
            p.add_argument("--out-dir", help="default: <train.out_dir>")
            p.add_argument("--points-dir", help="share prediction points between models: read them from here, or draw "
                           "and write them here first (they name patients, so keep them with the data)")
            p.add_argument("--targets", help="a file of target names, one per line (e.g. configs/targets/delphi2m.txt): "
                           "also report the means over those targets only")
            p.add_argument("--query", choices=["event", "gap"], default="event",
                           help="read the model at the event a point predicts from, or at a \"no event\" token "
                           "placed one gap before the diagnosis (models with that token only)")
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
        if args.source == "toy":
            from modlehr.data.toy import convert
            print(convert(args.out, n=args.subjects))
            return
        if not args.raw_dir:
            raise SystemExit(f"convert {args.source} needs --raw-dir")
        if args.source == "synthea":
            from modlehr.data.adapters.synthea import convert
            print(convert(args.raw_dir, args.out))
        elif args.source == "mimic-iv-core":
            from modlehr.data.adapters.mimic_iv_core import convert
            print(convert(args.raw_dir, args.out, diagnosis_time=args.diagnosis_time,
                          extra=tuple(e.strip() for e in args.extra.split(",") if e.strip())))
        else:
            from modlehr.data.adapters.mimic_iv_note import convert
            if not args.meds_dir:
                raise SystemExit("mimic-iv-note needs --meds-dir, the MIMIC-IV MEDS dataset to add notes to")
            print(convert(args.raw_dir, args.meds_dir, args.out, tuple(k.strip() for k in args.note_kinds.split(","))))
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
    elif args.command == "export-delphi":
        from modlehr.train.delphi import export
        export(cfg, args.out or f"{cfg.train.out_dir}_delphi_data", args.delphi_out or f"{cfg.train.out_dir}_delphi_code")
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
    elif args.command == "delphi":
        from modlehr.evaluate.delphi import delphi_auc
        for gap in (float(g) for g in args.gap_days.split(",") if g.strip()):
            suffix = "" if args.query == "event" else f"_{args.query}"
            out = Path(args.out_dir or cfg.train.out_dir) / f"evaluate_delphi_gap{gap:g}d{suffix}.csv"
            delphi_auc(cfg, checkpoint, gap, out, seed=cfg.train.seed,
                       points_dir=args.points_dir, query=args.query, targets_file=args.targets)
    elif args.command == "simulate":
        from modlehr.generate.simulate import simulate
        simulate(cfg, checkpoint, args.start_age, args.subjects, args.out_dir, seed=cfg.train.seed)
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
