---
title: "Wiki Log"
type: synthesis
created: 2026-04-15
updated: 2026-04-15
status: review
author: llm
---

# Wiki-Log

## [2026-04-15] meta | Projekt initialisiert
- Ordnerstruktur angelegt
- Schema-Dokument (CLAUDE.md) erstellt
- Index und Log initialisiert

## [2026-11-17] plan | gray-zone-triage pre-registered
- Design approved: input/ideas/gray-zone-triage-design.md
- Plan pre-registered: input/ideas/gray-zone-triage-plan.md (H1: R >= 0.20, alpha = beta = 0.05)
- Known gap at pre-registration: ICE-ID companion methods paper not retrieved (search credits exhausted, arXiv blocked by proxy)

## [2026-11-17] data | gray-zone-triage Task 1 done
- ICE-ID raw files fetched and fingerprinted (input/data/README.md); row count and label rate match the dataset paper
- Session network switched to the saved `record-linkage-commercial` environment to reach huggingface.co

## [2026-11-17] review | gray-zone-triage Task 2 reviews
- Spec review: PASS (independent rebuild reproduces all 1,194,338 pairs and the split). Format differs from plan: `candidates.csv` + `records.jsonl` joined on rid instead of one `candidates.jsonl` with inline records (storage only; inline records would be several GB). Script appends large derived files to `.gitignore`.
- Edge case: one record has raw sex `KArl`; the case-sensitive exp-1 SEX map leaves it unblocked (1 anchor, 1 true pair lost).
- Quality review: SOUND WITH ISSUES. Verified in main context:
  - 2026-11-17 · deviation-candidate · plan=gray-zone-triage · `partner` is the spouse's expert-curated `person` id (99.67% equal to person of the `source_partner` row; filled for 13.2% of labelled vs 0.1% of unlabelled records). B2's cross-wave partner rule therefore reads the gold-standard linkage of the spouse. Fires on 2.1% of candidate pairs, 10% of positives. `farm` is a place id (filled for 99.8% of unlabelled records), not leakage.
  - Experiment 1 impact: 4 of 200 pairs had a partner match; B2 AUC without partner 0.900 [0.856, 0.937] vs 0.905; KILLED-AT-4 unchanged.
- No thresholds computed, no LLM run, no test outcome inspected yet. Awaiting researcher decision before Task 3.

## [2026-11-17] amendment | gray-zone-triage Amendment 1 (pre-data)
- 2026-11-17 · deviation · plan=gray-zone-triage · Researcher approved Amendment 1 (see plan): partner_id blanked in all inputs (gold-standard leakage), bootstrap unit = anchor persons, added name-block CI / duplicate-key / anchor-weighted sensitivity, end-to-end misses and per-wave descriptives, case-insensitive sex, storage format recorded. H1 and frozen parameters unchanged. No threshold, LLM output or test outcome existed at the time.
- Task 2 re-run: 1,194,339 pairs (+1 from `KArl`), split unchanged; partner rule would have fired on 24,908 pairs (2.09%), 10.12% of positives. All unit checks pass.

## [2026-11-17] review | gray-zone-triage Task 3
- Spec review (subagent): PASS. Independent refit reproduces sample, thresholds and region counts; coefficients differ at the 5th decimal (sklearn lbfgs tolerance), region membership identical. Later tasks apply the saved b0/b1 (equivalently the saved B2 cut-offs), never a refit.
- Quality review: subagent stopped by the researcher; review done in the main context instead, at the researcher's request. Findings, none changing H1 or frozen values:
  - Optimism: thresholds are set and checked on the same calibration sample; the calibration missed-match rate is exactly 0.0500, so the realized test rate can exceed beta and trigger criterion 4 regardless of the LLM. Reported as a risk, not corrected.
  - Small gray zone: 329 calibration gray pairs, about 79 negatives, so LLM specificity (and the strength of the Bayesian update) is estimated imprecisely. Report the sens/spec CIs in Task 7.
  - Operating point: about 22% of calibration positives fall in auto-reject at beta = 0.05 (beta bounds the missed-match share among rejected pairs, not recall). Report recall of auto-decisions and end-to-end misses explicitly (Amendment 1, item 4).
- From Task 4 on, reviews are done in the main context and recorded here.

## [2026-11-17] data | gray-zone-triage Task 4
- Calibration gray zone rebuilt with saved b0/b1: 329 pairs, equal to thresholds.json; all used (cap 800 not reached).
- Test sample: 5,112 anchor records (4,895 anchor persons), 22,258 pairs; regions accept 2,838 / reject 18,820 / gray 600 (L_B2 = 0.027). Selection used B2 scores and frozen thresholds only.
- Review (main context): gray files carry only pair_id, a, b (no labels, partner_id blank); every taken anchor complete; gray_test equals the sample's gray region; no overlap between calibration and test gray pairs. 78% of test pairs come from waves 1835-1855 (descriptive, follows the name-frequency distribution).
- `labels_test.jsonl` is a mechanical copy for Task 7; no statistic on test labels has been computed.
