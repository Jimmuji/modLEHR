# Results so far

Back to the [README](../README.md).

## MIMIC-IV diagnoses

MIMIC-IV 3.1 patients, admissions and diagnoses (`modlehr convert mimic-iv-core`): 364,627 patients.
Targets are ICD-10 three-character categories, with ICD-9 codes mapped through the CMS GEMs: 1,017
diagnoses and death. Landmarks 40 to 80, a 2-year horizon, 1,062 (target, landmark) pairs on 73,083
held-out patients. Each transformer is 256 wide with 6 layers and 512-event windows, trained for
20,000 steps at batch 64 with seeds 1 to 5 on a cluster, one run per 10G H100 slice
(about 30 minutes each). Risk is each target's own probability within the horizon
(`evaluate.risk: marginal`). Differences are paired over the 1,062 pairs; the interval is over
seeds (`modlehr compare`), which is the one that decides whether two models differ.

| Run | Parameters | Mean AUC (range over seeds) | Difference, 95% CI over seeds |
|---|---|---|---|
| Plain GPT | 5.34M | 0.676 (0.676 to 0.677) | |
| Rotary over age (`rope_age`) | 5.34M | 0.677 (0.676 to 0.677) | +0.0006 (−0.0001 to +0.0012) |
| RMSNorm (`rmsnorm`) | 5.34M | 0.676 (0.676 to 0.677) | +0.0001 (−0.0008 to +0.0010) |
| Grouped-query attention (`gqa`) | 4.94M | 0.676 (0.676 to 0.677) | +0.0000 (−0.0008 to +0.0008) |
| Query and key normalization (`qk_norm`) | 5.34M | 0.676 (0.676 to 0.677) | +0.0000 (−0.0009 to +0.0010) |
| ALiBi over age (`alibi_age`) | 5.34M | 0.676 (0.676 to 0.677) | −0.0001 (−0.0012 to +0.0010) |
| SwiGLU feed-forward (`swiglu`) | 5.33M | 0.676 (0.676 to 0.676) | −0.0001 (−0.0008 to +0.0006) |
| Sliding window over 5 years (`local_age`) | 5.34M | 0.676 (0.674 to 0.677) | −0.0003 (−0.0020 to +0.0013) |
| Multi-head latent attention (`mla`) | 4.85M | 0.676 (0.675 to 0.677) | −0.0003 (−0.0007 to +0.0001) |
| Mixture of experts (`moe`) | 8.48M | 0.676 (0.675 to 0.676) | −0.0004 (−0.0011 to +0.0004) |
| Logistic regression, same inputs | | 0.653 | −0.0231 (−0.0237 to −0.0226) |

How to read it:

- **On real records the transformer beats the classical model** by 0.023, steadily over five seeds;
  on Synthea, below, it was the reverse. Under Delphi-2M's controls (`controls: never`) the plain
  GPT scores 0.665 and logistic regression 0.632.
- **Parts from recent language models do not move AUC here.** No difference reaches 0.001, and every
  interval includes zero. With 1,018 targets and 3.6M training events, the architecture is not
  what limits this task.
- **Calibration is off** (expected over observed new diagnoses 1.84 to 1.95 over these seeds; 1.59 on
  the October 19 reruns after the second review, 3.7 at age 40 and 1.0 at 80). The head holds each
  hazard constant from the last event before the landmark, often a discharge years earlier, when the
  risk of coming back is highest. Delphi-2M's "no event" tokens alone make it worse (2.9), since with
  follow-up from birth they fill the years before the first admission; with follow-up from the first
  visit they bring it to 1.16 (`mimic_iv_core_gpt_no_event_first_contact.yaml`, three seeds).
- **The lead over logistic regression is in every ICD-10 chapter** (+0.005 to +0.037) and grows with
  age, from +0.003 at landmark 40 to +0.036 at 70: longer histories are where the sequence model
  helps (`scripts/analysis/by_chapter.py`, October 19 runs).
- **These numbers replace an earlier set.** An external review found that only the first of several
  targets recorded at the same time was a training label (85% of first diagnoses never were), that
  outpatient BMI extended follow-up, that calibration dropped censored patients and that the
  intervals ignored seed-to-seed variation (see the [changelog](../CHANGELOG.md)). Fixing them raised the plain GPT from
  0.665 to 0.676 and its lead over logistic regression from 0.012 to 0.023; the old score,
  `risk: next_event`, gives 0.670 on the fixed models.

## MIMIC-IV diagnoses plus BMI

BMI from MIMIC-IV's outpatient record (`hosp/omr`, added by `convert mimic-iv-core` when the table is
there): 1.90M readings for 153,725 of the 364,627 patients, one per year kept, `follow_up: false`
(an outpatient reading is an input; diagnoses are recorded only at admissions), so cases and
controls are the same patients as above in all 1,062 pairs. Same model and seeds. This is the
closest MIMIC-IV has to Delphi-2M's lifestyle inputs, which it reads as three levels.

| Inputs | Mean AUC | vs diagnoses only | vs BMI as an event | Logistic regression |
|---|---|---|---|---|
| Diagnoses only | 0.676 | | | 0.653 |
| + BMI as an event, without its value (`value_mode: none`) | 0.701 | +0.0249 (+0.0240 to +0.0258) | | 0.676 |
| + BMI in 3 bins (`value_mode: binned`) | 0.701 | +0.0249 (+0.0244 to +0.0254) | −0.0000 (−0.0011 to +0.0010) | 0.674 |
| + BMI as a number (`continuous`, `value_encoding: mlp`) | 0.702 | +0.0253 (+0.0249 to +0.0256) | +0.0004 (−0.0006 to +0.0013) | 0.676 |

- **BMI adds 0.025, and the value does not matter.** Bins, a number and no value at all score the
  same. The gain comes from having an outpatient reading, which marks a patient seen in the
  hospital's clinics, not from the reading itself.
- **The transformer keeps its lead with BMI** (0.701 against 0.676 for logistic regression, a gap of
  0.025, 0.024 to 0.026 over seeds).

## Synthea

Synthea, 22,962 patients; 164 targets; landmarks 40 to 70; 5-year horizon; 124 (target, landmark)
pairs scored on 4,564 held-out patients. Each transformer has 1.25M parameters (GQA 1.16M) and
3,000 training steps, and is trained with seeds 1, 2 and 3; each pair's AUC is averaged over the
seeds before comparing. The difference is the mean paired AUC difference against the plain GPT,
with its 95% interval.

| Run | Mean AUC | Difference | Held-out NLL |
|---|---|---|---|
| Plain GPT (binned values) | 0.590 | | 4.778 |
| Rotary over age (`rope_age`) | 0.594 | +0.004 (+0.000 to +0.008) | 4.773 |
| ALiBi over age (`alibi_age`) | 0.588 | −0.002 (−0.006 to +0.001) | 4.761 |
| Grouped-query attention (`gqa`) | 0.589 | −0.001 (−0.005 to +0.003) | 4.770 |
| Values read directly (`value_encoding: mlp`) | 0.588 | −0.002 (−0.006 to +0.003) | 4.771 |
| Logistic regression, same inputs | 0.612 | +0.022 (+0.011 to +0.034) | |
| Gradient boosting, same inputs | 0.606 | +0.016 (+0.005 to +0.027) | |

The newer parts, one seed each so far (seed 1, against the plain GPT's seed 1, which scores 0.589
and 4.781):

| Run | Mean AUC | Difference | Held-out NLL |
|---|---|---|---|
| Query and key normalization (`qk_norm`) | 0.594 | +0.005 (−0.002 to +0.011) | 4.754 |
| SwiGLU feed-forward (`swiglu`) | 0.592 | +0.003 (−0.003 to +0.010) | 4.748 |
| RMSNorm (`rmsnorm`) | 0.591 | +0.002 (−0.003 to +0.007) | 4.773 |
| Mixture of experts (`moe`, 2.03M parameters, same active width) | 0.588 | −0.001 (−0.012 to +0.009) | 4.737 |
| Sliding window over 5 years (`local_age`) | 0.585 | −0.004 (−0.015 to +0.006) | 4.812 |
| Multi-head latent attention (`mla`, 1.13M) | 0.583 | −0.006 (−0.016 to +0.004) | 4.801 |

None differs from the plain GPT beyond its interval; `scripts/compare_synthea.sh` runs all of them over
three seeds.

How to read it:

- **Only rotary attention over age moves AUC**, and only just: +0.004, with the interval touching
  zero. With one seed the same comparison gave +0.007; averaging seeds roughly halved it, which is
  why `modlehr compare` averages them.
- **The likelihoods do not separate the variants.** Across the three seeds of one variant the
  held-out NLL varies by up to 0.03; the differences between variants are 0.01 to 0.02.
- **The classical models beat every transformer.** At 3,000 steps the transformers are
  undertrained, so architecture comparisons should wait until the baseline at least matches the
  logistic regression.
- **Simulated futures miss outcomes, which AUC does not show.** The number of next outcomes is
  close to right (expected over observed 0.94 to 0.98 over the three seeds), but futures sampled
  from age 50 hold 0.67 of the observed outcomes per person-year, with too many deaths.
- **Synthea limits what can be learned.** It generates diseases from a small number of rules, so
  the signal it holds is limited. It is the place to check that the pipeline runs, not where an
  architecture should be judged; model comparisons belong on real records (MIMIC-IV).
