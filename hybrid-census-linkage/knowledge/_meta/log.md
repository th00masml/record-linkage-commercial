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

## [2026-11-17] analysis | gray-zone-triage Task 5 (Fellegi-Sunter, Splink 5.0.0)
- Implementation decisions taken before any outcome (plan did not specify them):
  - Two EM passes, because Splink cannot estimate m for a field in the EM blocking rule: pass 1 blocks on name, patronym, sex, adjacent wave, same parish (trains birth year); pass 2 blocks exactly like the candidates (trains parish, county, farm, distance).
  - Prior: the frozen rule is D_FS = 1 if match probability >= 0.5, which depends on the prior. Splink's default prior (1e-4 per random pair) would put almost every candidate pair below 0.5. The prior used is the label-free EM fixed point on the 594,131 calibration candidate pairs: lambda_b = 0.1762 (true calibration base rate, not used: 0.189).
- Result: all levels trained; 929 gray pairs scored; score_pair agrees with predict() on calibration gray pairs.
- Finding (no labels used): inside the gray zone p_FS has median 1.0 and minimum 0.62, so D_FS = 1 for all 329 calibration and 600 test gray pairs. The binary decision is constant, its likelihood ratio will be about 1 and R_FS will be about 0 by construction. H2 as registered is therefore a weak comparator.
- 2026-11-17 · note · plan=gray-zone-triage · Registered H2 kept unchanged. Task 7 adds an EXPLORATORY H2 variant with the D_FS threshold chosen on the calibration gray zone; it is reported as exploratory and does not enter the verdict. The same degeneracy can affect the LLM decision; under the registration that would show up as R_LLM near 0 and a KILLED verdict, which is the honest outcome.
- Review (main context): spec matches Task 5 and Amendment 1 (partner comparison dropped, u by random sampling, m by EM on calibration candidates only, D_FS at 0.5, outputs written). Quality notes: u uses 1e6 random pairs (Splink suggests more; no level untrained); the constant D_FS is the substantive issue above.

## [2026-11-17] analysis | gray-zone-triage Task 7 script ready, not run on test
- `output/data-analysis/analyze.py` implements the registered decision rule, criteria 1-5, H2, Amendment 1 sensitivity and descriptive outputs, the exploratory Youden-threshold H2 (Task 5 log), and the git commit-order verification.
- Test labels stay sealed until `raw_qwen_gray.jsonl` is committed: computing B2 or FS on the test split earlier would reveal criteria 3/4 (they depend on B2's auto regions) before the LLM run.
- Dry run (`--dry-run`: calibration sample plays "test", random decisions play the LLM, nothing written): pipeline runs end to end in 8 s. FS gives R = 0 as predicted (constant D_FS). Random decisions give R = 0.09 and missed-match 0.052 > beta, i.e. noise alone moves borderline pairs and the beta guard catches it; part of that R comes from estimating sens/spec on the same pairs in the dry run.
- Review (main context): definitions checked line by line against Hypothesis, Falsification Criteria, Frozen parameters and Amendment 1.
