---
title: "Does routing only a rule's uncertain band to a local LLM reduce clerical review in Icelandic census linkage at a fixed false-match rate?"
type: design
created: 2026-11-17
updated: 2026-11-17
tags: [entity-resolution, record-linkage, historical-census, iceland, llm, fellegi-sunter, hybrid]
status: draft
author: mixed
---

# Gray-zone LLM triage on top of a hand-built census linkage rule

## Research Question

On adjacent-wave homonym candidate pairs from the Icelandic census (ICE-ID),
does sending only the pairs in the uncertain band of the calibrated B2 rule to
a local LLM (qwen2.5:14b), and combining the two by Bayesian update (Ather
2026), reduce the share of pairs that need clerical review, at a fixed
false-match rate, compared with B2 alone and with a non-LLM second stage
(Fellegi–Sunter via Splink) used in the same way?

The question can be answered "no", and a "no" is informative: it would extend
experiment 1's KILLED-AT-4 verdict from "LLM as matcher" to "LLM as triage".

## Relevance / State of the field (brief)

Experiment 1 in this repo found a hand-built rule (B2, AUC 0.905) beats both
claude-sonnet-5 (0.816) and qwen2.5:14b (0.787) as a standalone matcher on
200 balanced ICE-ID pairs. The literature agrees for clean structured fields:
on North Carolina voter records, fastLink and Splink tie an LLM-labelled
linker at about 97 F1 (Dasanaike 2026). The one published gray-zone hybrid
(Ather 2026, Statistical Journal of the IAOS) cut the clerical-review band
from 44.8% to 21.5% of pairs, but on Canadian business names, where the
signal is textual and LLMs have a known edge. Gap: whether that gain carries
over to person records whose signal is birth year, parish, farm and partner
rather than text. See `../../../research/llm-vs-rules-er-brief.md` for
sources; full literature work belongs to `literature-review`, which must also
locate the ICE-ID companion methods paper before pre-registration.

## Method

Quantitative, pre-registered. B2 code (`src/baseline.py`) and the experiment 1
prompt (`src/score.py`) are frozen. Any prompt variation is allowed only on
the calibration split.

1. Calibrate the B2 score to a match probability with logistic regression on
   the calibration split. (On the 200 used pairs, B2's mixed band was roughly
   2.9–5.0 points, dominated by ties at 4.0 and 5.0. Those pairs motivate the
   design but are excluded from all new data.)
2. On the calibration split, choose a lower threshold T_L and an upper
   threshold T_U giving the widest auto-decision region in which the
   false-match rate among auto-accepted links is at most alpha and the
   missed-match rate among auto-rejected pairs is at most beta. Pairs with
   T_L <= p < T_U form the gray zone.
3. Send gray-zone pairs to qwen2.5:14b. Its binary decision D comes from token
   log-probabilities, P(yes) >= 0.5, where the local Ollama version exposes
   them; otherwise from the majority of k samples. Estimate the LLM's
   sensitivity and specificity on calibration gray-zone pairs.
4. Combine prior and D by Bayesian update (Ather 2026). Apply the same
   thresholds to the posterior. Pairs still in the band go to clerical review.
5. Repeat steps 3–4 with Fellegi–Sunter (Splink, unsupervised EM, same
   fields) in place of the LLM. This separates "an LLM helps" from "any
   second stage helps".

Primary metric: difference, in percentage points, in the share of test pairs
routed to clerical review, hybrid (LLM) minus B2 alone, with a bootstrap CI
clustered by anchor record.

Guard: the hybrid counts as a success only if its realized false-match rate
among auto-accepted test links does not exceed alpha. The Bayesian update
assumes the LLM's decision is conditionally independent of B2 given the true
label. Here the LLM sees the same fields as B2, so the posterior will tend to
be overconfident, and the guard is what catches that.

Secondary: the same metric for the Fellegi–Sunter second stage, and the
LLM-vs-Fellegi–Sunter difference within the gray zone.

Numerical values for alpha, beta, k, the minimum required reduction (the
KILLED threshold) and the sample size come from `writing-research-plan`,
fixed before any LLM run.

## Data Sources / Material Basis

- ICE-ID (de Carvalho et al. 2025), Hugging Face `goldpotatoes/ice-id`, CC-BY 4.0:
  984,028 Icelandic census records, 1703–1920, 226,864 expert-curated person
  identifiers. Files: people.csv, parishes.csv, counties.csv.
- Candidate generation with a realistic ratio: for each record in wave t,
  every record in the adjacent wave t+1 with the same first name, patronym
  and sex and a birth year within ±2. Label = same expert identifier. No
  rebalancing.
- Exclusion: every record appearing in experiment 1's 200 pairs.
- Split: calibration vs. test by anchor person, so no person appears on both
  sides. The split ratio and target size come from the plan's power analysis,
  aiming for several hundred gray-zone pairs in the test split.

## Output Target

☐ Book chapter   ☒ Journal article   ☐ Grant proposal   ☐ Talk   ☐ Other

One methodological article combining experiment 1 (LLM as matcher,
KILLED-AT-4) and this experiment (LLM as triage). Candidate venues: Historical
Methods, or the Statistical Journal of the IAOS, where Ather 2026 appeared.

## Expected Results

- Positive: the hybrid reduces clerical review by at least the pre-registered
  margin versus B2, also beats the Fellegi–Sunter second stage, and keeps the
  realized false-match rate within alpha. Reading: an LLM adds value on
  structured person records, but only as triage of the uncertain band.
- Mixed: the hybrid beats B2 but not the Fellegi–Sunter second stage. Reading:
  the gain comes from having a second stage, not from the LLM.
- Null (KILLED): no reduction beyond the margin, or the guard fails. Reading:
  experiment 1's verdict extends to triage; on clean census fields the LLM is
  not worth its cost even when confined to the gray zone.

## Ethical / Methodological Risks

- Ethics: records from 1703–1920; all persons are deceased. Data are CC-BY
  and processed on local machines only. Minimal risk; attribution required.
- Label noise in the expert identifiers attenuates every difference. It cannot
  be removed, so it must be reported.
- Dependence between the LLM's judgment and B2 (same fields) inflates
  posterior confidence. Mitigated by the realized false-match guard, not by
  modelling.
- A narrow gray zone could leave the test underpowered. Handled by the plan's
  power analysis and sample size.
- Ollama log-probability support depends on the installed version (not
  verified); fallback is k-sample majority, pre-specified.
- The ICE-ID companion methods paper may already report something similar;
  to be checked in `literature-review` before pre-registration.

## Feasibility

- In this cloud container (CPU only): data download, candidate generation,
  B2 and Fellegi–Sunter calibration, threshold selection, analysis.
- On the researcher's machine: qwen2.5:14b scoring via Ollama, the only
  GPU-bound step. Expect hours for several thousand calls with k samples.
- Total: a few working days, most of it the LLM run. No collaborators needed.
- Tools: Python, numpy, splink, Ollama; existing `src/` code reused unchanged.

## Scope Boundaries (what is NOT investigated)

Not investigated: messy or OCR-garbled fields, the "select among homonym
candidates" format (each gets its own design doc), Claude or any commercial
model as an instrument (exploratory side note at most, not pre-registered),
links across non-adjacent waves, and data from other countries.

## Next Step

Hand off to `writing-research-plan` to produce
`input/ideas/gray-zone-triage-plan.md` with frozen alpha, beta, k, KILLED
margin and sample size (status=pre-registered).
