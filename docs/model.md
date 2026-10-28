# The model

Back to the [README](../README.md).

```
tokens, ages, values → input head → N transformer blocks → final norm → output head
```

**Input head.** Each event's vector is `E_code(c) + E_age(t) + m · F(z) + T(words)`: a learned
embedding of its token, an encoding of the patient's age at the event (in place of a position),
when the event has a value and the value encoding is on, a small network applied to the scaled
value, and when the event is a note and the text encoding is on, an encoding of its words. An
optional `extra` tensor is added here too; it is where images will plug in.

**Blocks.** Pre-norm transformer blocks with causal attention: an event attends to itself and
to earlier events, never to later ones, and never to padding.

**Output head (`competing_risk`).** One rate per target per year. Between events each target k
has a constant rate λ_k; the next target is k after Δt years with likelihood
λ_k · exp(−Λ Δt), where Λ is the sum of all rates, and a stretch with no target until the record
ends has likelihood exp(−Λ Δt). Several targets recorded at the same time (a set S) have
likelihood ∏_{k in S} λ_k · exp(−Λ Δt). The loss is the negative log of these. With one target at
a time it is the same likelihood as Delphi-2M's next-event and waiting-time losses (Shmatko et
al., *Nature* 2025), written as one expression, with the end of follow-up handled as censoring.
Rates start at each target's training-set rate. The risk of k within H years, the score AUC ranks
by, is 1 − exp(−λ_k H) (`evaluate.risk: marginal`); `next_event` gives (λ_k / Λ)(1 − exp(−Λ H)),
the probability that k is the next target.

Every part is registered by name and picked in `configs/model/`:

| Kind | Registered | Notes |
|---|---|---|
| `attention` | `mha`, `gqa`, `rope`, `rope_age`, `alibi_age`, `mla`, `local_age` | multi-head (GPT-2); grouped-query (Llama, Qwen); rotary over event order; rotary over age; ALiBi over age (BLOOM, MPT use it over positions); multi-head latent attention, keys and values from a low-rank latent (DeepSeek-V2, V3); a sliding window over the last N years (Mistral and Gemma slide over positions). Any of them takes `qk_norm: true`, query and key normalization (OLMo 2, Gemma 3, Qwen3) |
| `norm` | `layernorm`, `rmsnorm` | |
| `ffn` | `gelu_mlp`, `swiglu`, `moe` | GELU MLP (GPT-2); gated SwiGLU (Llama); mixture of experts, each event routed to 2 of 4 SwiGLU experts with a load-balancing loss (Mixtral, DeepSeek, Qwen3-MoE). With the defaults an event passes through as many weights as the GELU MLP. All experts are computed and the ones not chosen weighted zero (`dispatch: dense`, the same result with fixed tensor shapes), so training costs about twice the MLP's compute; `dispatch: sparse` computes only the chosen ones |
| `time_encoding` | `sinusoidal_age`, `none` | |
| `value_encoding` | `none`, `mlp` | `mlp` reads values kept with `value_mode: continuous` |
| `text_encoding` | `none`, `hashed_bag` | `hashed_bag` averages learned embeddings of a note's words, hashed into 65,536 buckets with no external tokenizer |
| `head` | `competing_risk` | |
| `target_grouping` | `identity`, `icd10_3char`, `icd_3char`, `icd_gems`, `phecode` | `icd_3char` handles ICD-9 and ICD-10 mixed, as in MIMIC-IV, keeping ICD-9 categories apart; `icd_gems` maps ICD-9 through the CMS General Equivalence Mappings first, so both coding eras share one target set; `phecode` maps both to phecodes (PheWAS catalog). The last two read public mapping files named in `data.target_grouping_options` |

Options go in the config, for example `attention_options: {n_kv_heads: 2}` or
`ffn_options: {n_experts: 8}`. Each part has a ready model config in `configs/model/`. All keys and
defaults: [docs/configuration.md](configuration.md).

Every file in `configs/model/` other than `gpt_baseline.yaml` lists only the part it swaps, so it
layers onto a model of any size. `configs/experiment/mimic_iv_core_gpt_rope_age.yaml` is
`defaults: [mimic_iv_core_gpt.yaml, ../model/gpt_rope_age.yaml]`: the 256-wide MIMIC model with
rotary attention over age.
