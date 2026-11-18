"""examples/prepared_data.py reads the prepared data without modLEHR; it must read what modLEHR itself reads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from modlehr.prepare.dataset import open_split, random_windows
from modlehr.prepare.pipeline import run_prepare
from modlehr.train.finetune import history, task_samples, write_samples
from test_finetune import write_labels

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from prepared_data import PreparedData  # noqa: E402

QUIET = lambda *_: None  # noqa: E731


def test_standalone_reader_reads_what_modlehr_reads(tiny_config, tmp_path):
    run_prepare(tiny_config.data, log=QUIET)
    tiny_config.finetune.labels = str(write_labels(Path(tiny_config.data.meds_dir), tmp_path / "labels" / "a60", 60.0))
    write_samples(tiny_config, log=QUIET)
    data = PreparedData(tiny_config.data.prepared_dir)
    split = open_split(tiny_config.data.prepared_dir, "train")
    assert len(data.split("train")) == len(split)

    ours = data.split("train").windows(16, 8, np.random.default_rng(3))
    theirs = random_windows(split, 16, 8, np.random.default_rng(3))
    n = ours["tokens"].shape[1]
    np.testing.assert_array_equal(ours["tokens"], theirs["tokens"][:, :n].numpy())
    assert not theirs["tokens"][:, n:].any()  # modLEHR pads further, to a multiple of 64
    np.testing.assert_array_equal(ours["ages"], theirs["ages"][:, :n].numpy())
    np.testing.assert_array_equal(ours["mask"], (theirs["tokens"][:, :n] != 0).numpy())

    assert data.tasks() == ["a60"]
    task, samples = data.task("a60", "train"), task_samples(tiny_config, "train", QUIET)
    np.testing.assert_array_equal(task["label"], samples["label"])
    rows = np.arange(len(task["label"]))
    batch = data.task_inputs(task, "train", 8, rows)
    for r in rows:
        if samples["n"][r] == 0:
            assert not batch["mask"][r].any()
            continue
        expected = history(split, samples["index"][r], samples["n"][r], 8)["tokens"]
        np.testing.assert_array_equal(batch["tokens"][r][batch["mask"][r]], expected)
