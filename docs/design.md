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

## Input and target vocabularies

Inputs can be as fine as the source records. Targets are kept small: disease codes are mapped
by a registered grouping (`identity`, `icd10_3char`). Delphi predicts about 1,300 ICD-10
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

## Labels and the loss

Each event is labeled with the next target event strictly later in time (its target id and the
years until it), or censored at the end of the record. Only the last event at each distinct
time carries a loss, so a prediction is made after everything recorded at that moment.

The `competing_risk` head gives each target a log-rate per year. Between events the hazards are
constant, so the next target is k after dt years with likelihood `lambda_k exp(-Lambda dt)`, and
a censored stretch has likelihood `exp(-Lambda dt)`, with `Lambda = sum_k lambda_k`. This equals
Delphi's next-token cross-entropy plus its exponential waiting-time loss, written as one
likelihood, with censoring handled directly instead of through padding tokens. Log-rates start
at the training-set rate of each target.

Risk of target k within a horizon H: `(lambda_k / Lambda) (1 - exp(-Lambda H))`.

## Evaluation

For each landmark age L and target k: inputs are all events up to L; a case has its first k in
(L, L + H]; subjects who already had k are left out, and so are subjects whose record ends
before L + H without k. AUC per (target, landmark) with at least `min_cases` cases and
controls; the summary is the mean over those pairs.

## Model

`input head → N blocks → final norm → head`, assembled from registered parts:

- attention: `mha` (causal, key padding masked)
- norm: `layernorm`, `rmsnorm`
- ffn: `gelu_mlp`, `swiglu`
- time encoding: `sinusoidal_age`, `none`
- value encoding: `none`, `mlp`
- head: `competing_risk`

The input head takes an optional `extra` tensor added to each event's embedding; that is where
other modalities (clinical notes, images) plug in later.

## Not yet

- other attention variants and positional schemes (GQA, MLA, linear attention, RoPE over time)
- adapters beyond Synthea (MIMIC-IV, eICU and EHRSHOT have existing MEDS ETLs)
- clinical notes through `text_value`, then images
- MEDS-DEV tasks and label files (`subject_id, prediction_time, boolean_value`)
- multiple seeds and confidence intervals in evaluation
