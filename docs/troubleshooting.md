# Troubleshooting

Back to the [README](../README.md).

**`import modlehr` fails after the editable install (macOS, Python 3.13).** The install's `.pth`
file is created with the hidden flag, and Python 3.13 skips hidden `.pth` files. Clear it:
`chflags nohidden .venv/lib/python3.13/site-packages/*.pth`. If it keeps coming back, run commands as
`PYTHONPATH=src .venv/bin/python -m modlehr ...`; the tests and `scripts/compare_synthea.sh` already do.

**`no target had enough cases and controls`.** Each (target, landmark) pair needs `min_cases` cases
and controls in the evaluation split. Lower `evaluate.min_cases`, add landmarks, or use a longer
horizon. Small datasets such as the MIMIC-IV demo need `min_cases` of 2 or so.

**`unknown attention 'x'; available: [...]`.** The name is not registered, or the module that
registers it is not imported; see [docs/extending.md](extending.md).

**A rerun of Synthea with the same seed gives a different population.** Synthea generates patients
on several threads, and in our runs the same seed produced 23,056 and then 22,962 patients. Keep the
converted MEDS directory rather than regenerating it, and compare models on the same copy.

**Memory.** `prepare` holds one MEDS shard at a time, plus the training values of every
measurement (pass 1). Peak memory on Synthea is 1.5 GB. If a source's shards are very large, reshard
it with the MEDS tools rather than narrowing the event rules.
