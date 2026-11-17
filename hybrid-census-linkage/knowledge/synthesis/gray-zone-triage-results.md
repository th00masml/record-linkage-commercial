---
title: "LLMs add little to structured census linkage, as matcher or as triage"
type: synthesis
created: 2026-11-17
updated: 2026-11-17
tags: [entity-resolution, record-linkage, historical-census, iceland, llm, hybrid]
status: review
author: llm
hypothesis: "H1 (gray-zone-triage plan): the B2 + qwen2.5:14b hybrid removes at least 20% of B2's clerical-review load at false-match and missed-match rates <= 0.05."
methodology: quantitative
---

# LLMs add little to structured census linkage, as matcher or as triage

## Research question

Can a large language model improve linkage of Icelandic census records
(ICE-ID) over a hand-built rule (B2), either as the matcher itself
(experiment 1) or confined to the rule's uncertain band (experiment 2,
`input/ideas/gray-zone-triage-plan.md`)?

## Line of argument

1. As a matcher, no. On 200 balanced homonym pairs, B2 reached AUC 0.905
   against 0.816 for claude-sonnet-5 and 0.787 for qwen2.5:14b; the frontier
   model was not significantly better than the local one (repo `RESULTS.md`).
   Removing the leaky `partner_id` field (found in experiment 2) changes B2
   to 0.900 and leaves the verdict intact.
2. As triage, barely. On 22,258 realistic candidate pairs, the hybrid removes
   4.7% [3.0%, 6.6%] of B2's review load against a pre-registered 20%
   (`output/data-analysis/RESULTS.md`). Verdict KILLED.
3. The mechanism is the same in both experiments. The disambiguating evidence
   is structured (birth year, parish, farm, distance). Where B2 is unsure, the
   LLM reading the same fields is unsure too, and it defaults to "same
   person": 96% of band pairs, specificity 0.11.
4. A plain probabilistic second stage does at least as well. Fellegi–Sunter
   with a calibrated threshold removed 9.0% of the load (exploratory), though
   it also missed the beta limit. The ceiling for any second stage on these
   fields looks low; the LLM sits below it.

## Position relative to the literature

This agrees with the clean-field results in `research/llm-vs-rules-er-brief.md`
(repo root): classical linkers tie or beat LLMs when records share clean
structured fields, and LLM gains come from textual variation the fields here
do not contain. Ather (2026) reported a halved gray zone on business names;
the present result suggests that gain does not transfer to person records
whose signal is structured.

## Open questions

- Messy fields (OCR-garbled names, free-text status) remain untested and are
  where the literature locates LLM gains: a separate design doc.
- The "select among homonym candidates" format, which gives the model record
  interactions, remains untested: a separate design doc.
- `partner` in ICE-ID is a curated person id; any published B2-style baseline
  on ICE-ID that uses it should be checked for the same leakage.
