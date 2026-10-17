"""Optional experiment tracking with Weights & Biases, on when `train.wandb_project` is set.

Training logs every check (train and tuning loss, the loss parts, learning rate, steps per second,
peak GPU memory), and wandb adds GPU utilization and memory over time on its own. `evaluate` and
`baseline` put their results (mean AUC, calibration, likelihood) in the run's summary. A run's wandb
id is kept in <out_dir>/wandb_id, so a resumed training run and its evaluation stay one wandb run.

Compute nodes without internet access: set WANDB_MODE=offline in the job, then upload from a login
node with `wandb sync <out_dir>/wandb/offline-run-*`.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from modlehr.config import Config


class Tracker:
    """A wandb run for one run directory, or nothing at all when tracking is off."""

    def __init__(self, cfg: Config, out_dir: str | Path, job_type: str):
        self.run = None
        if not cfg.train.wandb_project:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError("train.wandb_project is set but wandb is not installed: pip install wandb") from e
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        id_file = out_dir / "wandb_id"
        if not id_file.exists():
            id_file.write_text(uuid.uuid4().hex[:16])
        self.run = wandb.init(project=cfg.train.wandb_project, entity=cfg.train.wandb_entity or None,
                              name=out_dir.name, id=id_file.read_text().strip(), resume="allow",
                              dir=str(out_dir), job_type=job_type, config=cfg.to_dict())

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
