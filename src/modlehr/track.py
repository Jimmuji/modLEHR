"""Optional experiment tracking with Weights & Biases, on when `train.wandb_project` is set.

Training logs every check (train and tuning loss, the loss parts, learning rate, steps per second,
peak GPU memory), and wandb adds GPU utilization and memory over time on its own. `evaluate` and
`baseline` put their results (mean AUC, calibration, likelihood) in the run's summary. A run's wandb
id is kept in <out_dir>/wandb_id, so a resumed training run and its evaluation stay one wandb run.
The run's URL is printed when it starts.

Required tracking: with MODLEHR_REQUIRE_WANDB=1 in the environment (scripts/fir/env.sh sets it for every
Fir job), a training run (`run`, `train`, `finetune`) stops before its first step unless its wandb run
starts online: train.wandb_project set, WANDB_MODE not offline or disabled, logged in, wandb reachable,
and, when MODLEHR_WANDB_ENTITY is set, the run in that entity. Evaluation and baselines are not held to it.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from modlehr.config import Config

REQUIRED_FOR = ("train", "finetune")  # the job types MODLEHR_REQUIRE_WANDB=1 holds to online tracking


def wandb_required(job_type: str) -> bool:
    return os.environ.get("MODLEHR_REQUIRE_WANDB") == "1" and job_type in REQUIRED_FOR


class Tracker:
    """A wandb run for one run directory, or nothing at all when tracking is off."""

    def __init__(self, cfg: Config, out_dir: str | Path, job_type: str):
        self.run = None
        required = wandb_required(job_type)
        if not cfg.train.wandb_project:
            if required:
                raise RuntimeError("MODLEHR_REQUIRE_WANDB=1: set train.wandb_project, so that this training run "
                                   "is tracked in wandb")
            return
        mode = os.environ.get("WANDB_MODE", "")
        if required and mode in ("offline", "disabled", "dryrun"):
            raise RuntimeError(f"MODLEHR_REQUIRE_WANDB=1 but WANDB_MODE={mode}: training runs must log to wandb online")
        try:
            import wandb
        except ImportError as e:
            raise ImportError("train.wandb_project is set but wandb is not installed: pip install wandb") from e
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        id_file = out_dir / "wandb_id"
        if not id_file.exists():
            id_file.write_text(uuid.uuid4().hex[:16])
        try:
            self.run = wandb.init(project=cfg.train.wandb_project, entity=cfg.train.wandb_entity or None,
                                  group=getattr(cfg.train, "wandb_group", None) or None,
                                  name=out_dir.name, id=id_file.read_text().strip(), resume="allow",
                                  dir=str(out_dir), job_type=job_type, config=cfg.to_dict())
        except Exception as e:
            if required:
                raise RuntimeError(f"wandb could not start the run ({type(e).__name__}: {e}); not training "
                                   "without tracking (logged in? wandb reachable? entity?)") from e
            raise
        if required:
            self._check_online()
        url = getattr(self.run, "url", None)
        if url:
            print(f"[wandb] {job_type} run: {url}", flush=True)

    def _check_online(self) -> None:
        """Stop unless the run is online and, when MODLEHR_WANDB_ENTITY is set, in that entity."""
        run, want = self.run, os.environ.get("MODLEHR_WANDB_ENTITY")
        problem = None
        if run is None or getattr(run, "disabled", False):
            problem = "wandb tracking is disabled"
        elif getattr(run, "offline", False):
            problem = "the wandb run is offline"
        elif want and getattr(run, "entity", None) != want:
            problem = f"the wandb run went to entity {getattr(run, 'entity', None)!r}, not {want!r} (wandb login --relogin?)"
        if problem:
            if run is not None:
                run.finish(exit_code=1)
            self.run = None
            raise RuntimeError(f"MODLEHR_REQUIRE_WANDB=1: {problem}; not training without tracking")

    def log(self, values: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def summary(self, values: dict, prefix: str = "") -> None:
        """Flatten nested dicts into 'a/b' keys and put them in the run's summary."""
        if self.run is None:
            return
        for key, value in values.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                self.summary(value, prefix=f"{name}/")
            elif isinstance(value, (int, float, str, bool)) or value is None:
                self.run.summary[name] = value

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
