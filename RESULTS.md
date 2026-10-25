# Results

Run date: 2026-08-09. Instruments: `qwen2.5:14b` (local) and
`claude-sonnet-5` (commercial, via the authenticated Claude CLI, no API
key). Numbers recompute from cached outputs in `outputs/`.

## Verdict: KILLED-AT-4

A competent hand-built deterministic linker beats both LLMs on structured
Icelandic census linkage. The frontier model does not clear the bar, and
is not significantly better than the local 14B.

## The numbers

200 pairs: 100 same-person-across-adjacent-waves positives + 100 homonym
hard negatives (same first name + patronym + sex, born within 1 year,
different people). Names carry no signal for any pair.

| method | AUC | 95% CI |
|---|---|---|
| B1 birth-year only | 0.722 | [0.659, 0.787] |
| **B2 hand-built rule** | **0.905** | [0.862, 0.940] |
| M_local qwen2.5:14b | 0.787 | [0.728, 0.843] |
| M_comm claude-sonnet-5 | 0.816 | [0.758, 0.869] |

Primary (commercial vs best baseline):
- gain over B2 = **−0.089** (needed +0.05)
- paired ΔAUC (M_comm − B2) = **−0.089**, 95% CI [−0.137, −0.045],
  P(Δ≤0) = 1.000 — the rule is significantly better.
- VERDICT: **KILLED-AT-4**.

```
cd src
python prepare_pairs.py
python baseline.py            # B1, B2 (run first)
python score.py local && python score.py comm
python report.py
```

## Commercial vs local: not worth it

paired ΔAUC (M_comm − M_local) = +0.030, 95% CI [−0.037, +0.095],
P(Δ≤0) = 0.20. The frontier model is not significantly better than the
local 14B at this task.

## Why the rule wins

The disambiguating evidence here is structured: birth-year agreement,
parish/county, farm, partner id, and geographic distance from coordinates.
A weighted rule over those fields is exactly the right tool. The LLM reads
the same fields as prose and reasons about them, but adds no accuracy —
and sometimes over-thinks. On structured record linkage, decades of
deterministic/​probabilistic linkage methods are not beaten by dropping an
LLM on the raw fields.

Where an LLM might still help is the genuinely messy tail (OCR-garbled
names, inconsistent free-text, missing fields) — not tested here, and the
honest next step. On clean structured fields, use a rule.

## Memorization

Not applicable, by construction. The task links anonymized census rows
whose identity groupings are a private expert curation over internal ID
codes; there is no public key tying these individual records together for
a model to recall. The names are common Icelandic patronymics shared by
thousands. Scores reflect reasoning over the shown fields, not recall.

## Limits

- 200 pairs, one country, clean structured fields (the LLM's worst case
  relative to rules; its best case — messy free text — is untested).
- Positives are adjacent-wave links; harder long-gap links untested.
- CLI scores are single-sample (no temperature control); cached.
