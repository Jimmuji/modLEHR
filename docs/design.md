# Design notes

## Goal

A general framework for longitudinal EHR modeling: any source in, one format, one code base,
reproducible on public data. The framework is the contribution. Architectures from recent
LLMs are tested inside it by swapping registered parts; the first model is a plain GPT.

## Data contract: MEDS only

The core reads MEDS (meds 0.4): one row per event, `subject_id, time, code, numeric_value,
text_value`, plus codes metadata and subject splits. Everything specific to a source lives in
its adapter. MEDS fixes the storage format, not the vocabulary, so codes keep their source
coding (SNOMED CT, ICD-10, LOINC) and the choice of prediction targets is made in `prepare`.

## Time

MEDS stores datetimes; the model works in age. Age is event time minus the subject's
`MEDS_BIRTH`, in years. Static rows (no time) sit at age 0. Subjects without a birth event are
dropped. For sources with approximate birth dates (MIMIC-IV gives age at an anchor year), age
is approximate to about a year.

## Events and roles

A data config lists rules; each selects codes (prefix, list, description regex) and gives them
a role:

| Role | Meaning | Tokenization |
|---|---|---|
| target | an outcome the model predicts | target group (see below); first occurrence per subject by default |
| static | a fact without a time | the code, at age 0 |
| measurement | a numeric value | the code with its value, thinned to the first value per window |
| categorical | a text answer | `code//answer` |
| context | any other input | the code |
| text | a note | the code, plus its words for the text encoding |

Every rule also says whether its events extend follow-up (`follow_up`, true by default). Follow-up
ends at a subject's last such event, and no later than death. Outpatient measurements in a source
whose diagnoses come only from admissions are the case for `follow_up: false`: a reading after the
last admission must not make a subject look followed when no diagnosis could have been recorded.

## Input and target vocabularies

Inputs can be as fine as the source records. Targets are kept small: disease codes are mapped
by a registered grouping (`identity`, `icd10_3char`, `icd_3char`, `icd_gems`, `phecode`). Delphi predicts about 1,300 ICD-10
three-character codes; a source coded in SNOMED CT or full ICD-10-CM would otherwise produce
tens of thousands of targets. Target tokens are also input tokens, so sampled outcomes can be
fed back. Tokens seen in fewer than `min_subjects_per_token` training subjects are dropped.

## Values

`value_mode` decides how measurements enter:

- `binned`: the value becomes a token, `code//Q1..Qn`, cut at training-set quantiles. This is
  the plain GPT baseline: tokens only.
- `continuous`: the value is scaled per measurement (training mean and standard deviation,
  clipped to ±5) and read by the value encoding, `h_i = E_code(c_i) + E_age(t_i) + m_i F(z_i)`.
- `none`: the measurement is an event without a value.

## Text

A rule with role `text` keeps the code as a token and the words of `text_value`: lowercased, split
into runs of letters and digits, hashed with CRC-32 into `text_buckets` ids (the same on every
machine), the first `text_max_words` of each note. They are stored ragged per split (`text_ref`,
`note_offsets`, `note_words`), and the `hashed_bag` text encoding averages a learned embedding of
each word and adds it to that note's event. Datasets without text rules write no text arrays.

## Labels and the loss

Each event is labeled with the next time strictly later at which a target is recorded: every
target recorded then, and the years until it. In MIMIC-IV all diagnoses of an admission are timed
at its discharge, so most first diagnoses share their time with others, and each of them is a
label. If no target follows within follow-up, the event is censored at the end of follow-up.
Events after follow-up carry no loss. Only the last event at each distinct time carries a loss,
so a prediction is made after everything recorded at that moment.

The `competing_risk` head gives each target a log-rate per year. Between events the hazards are
constant, so the next target is k after dt years with likelihood `lambda_k exp(-Lambda dt)`, and
a censored stretch has likelihood `exp(-Lambda dt)`, with `Lambda = sum_k lambda_k`. Targets
recorded together (a set S) have likelihood `prod_{k in S} lambda_k exp(-Lambda dt)`. With one
target at a time this equals Delphi's next-token cross-entropy plus its exponential waiting-time
loss, written as one likelihood, with censoring handled directly instead of through padding
tokens. Log-rates start at the training-set rate of each target.

Risk of target k within a horizon H, the score AUC ranks by: `1 - exp(-lambda_k H)`, each target on
its own hazard (`evaluate.risk: marginal`), or `(lambda_k / Lambda) (1 - exp(-Lambda H))`, the
probability that k is the next target (`next_event`). Calibration uses the probability that k is
recorded within H, which means before death: `(lambda_k / (lambda_k + lambda_d)) (1 - exp(-(lambda_k
+ lambda_d) H))`.

## Evaluation

For each landmark age L and target k: inputs are all events up to L; a case has its first k in
(L, L + H]; subjects who already had k are left out, and so are subjects whose follow-up ends
before L + H without k (`evaluate.controls: at_risk`; `never` keeps them as controls when they
never have k, Delphi-2M's definition). AUC per (target, landmark) with at least `min_cases`
cases and controls; the summary is the mean over those pairs.

Calibration compares, at each landmark, the predicted number of targets first recorded in
(L, L + H] with the number observed, weighting the fully observed patients by the inverse
probability of still being followed (Kaplan-Meier of the follow-up end times), so patients whose
follow-up ends inside the window are not simply dropped. The task definition lives in one place,
`modlehr/evaluate/task.py`, and the model evaluation and the classical baselines both use it.

AUC is computed within an age group and only ranks patients, so it cannot see whether a model
predicts timing well; a time encoding or a different head can matter without moving AUC. The
evaluation therefore also reports the held-out negative log-likelihood of the training
objective, split into the event part and the time part.

Classical baselines (`modlehr baseline`: logistic regression and gradient boosting on token
indicators and latest values) run on the same pairs. They answer two questions before any
architecture comparison: has the transformer learned more than the inputs say on their own, and
is there signal beyond single features at all. On Synthea at 3,000 steps the logistic regression
still beats the transformer.

Comparisons (`modlehr compare`) average each pair's AUC over seeds, then take the paired
difference against a reference run over the pairs both scored, with a 95% interval over pairs and
one over seeds. Only the one over seeds includes how much a run moves with its seed.

`modlehr delphi` computes Delphi-2M's own AUC (by sex and age band, against patients who never
have the disease, predicted from just before the diagnosis), for comparison with its published
numbers.

## Model

`input head → N blocks → final norm → head`, assembled from registered parts:

- attention: `mha` (causal, key padding masked), `gqa`, `rope` (rotary over event order),
  `rope_age` (rotary over age), `alibi_age` (ALiBi penalty on years between events), `mla`
  (keys and values from a low-rank latent), `local_age` (a sliding window over years); any of
  them with `qk_norm`
- norm: `layernorm`, `rmsnorm`
- ffn: `gelu_mlp`, `swiglu`, `moe`
- time encoding: `sinusoidal_age`, `none`
- value encoding: `none`, `mlp`
- text encoding: `none`, `hashed_bag`
- head: `competing_risk`

The input head takes an optional `extra` tensor added to each event's embedding; that is where
other modalities (images) plug in later.

Sampling (`modlehr generate`) never draws a target already in the history or already drawn,
because targets are first occurrences.

## Not yet

- other attention variants (linear attention)
- more sources (eICU and EHRSHOT have existing MEDS ETLs; Synthea and MIMIC-IV are read today)
- images, through the input head's `extra` hook
- a pretrained clinical language model as a text encoding
- MEDS-DEV tasks and label files (`subject_id, prediction_time, boolean_value`)
