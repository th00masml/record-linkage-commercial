---
title: "Gray-zone LLM triage on top of a hand-built census linkage rule"
type: plan
created: 2026-11-17
updated: 2026-11-17
status: pre-registered
author: mixed
design: "[[gray-zone-triage-design]]"
---

# Gray-zone LLM triage on ICE-ID: Research Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `research-superpowers:executing-research-plan` to run this plan task by task. Tasks use checkbox (`- [ ]`) syntax for TodoWrite integration.

**Research Question:** On adjacent-wave homonym candidate pairs from ICE-ID, does sending only the pairs in the uncertain band of the calibrated B2 rule to qwen2.5:14b, combined by Bayesian update (Ather 2026), reduce the share of pairs routed to clerical review, at a fixed false-match rate, compared with B2 alone?

**Hypothesis (pre-registered, immutable):** H1: On the sealed test split, the B2 + qwen2.5:14b hybrid removes at least 20% of B2's clerical-review load (R >= 0.20) while keeping the realized false-match rate among auto-accepted links at or below 5% and the realized missed-match rate among auto-rejected pairs at or below 5%.

**Falsification Criteria:** H1 is rejected, verdict KILLED, if any of the following holds on the test split:
1. R < 0.20 (point estimate), or
2. the lower bound of the 95% cluster-bootstrap CI of R is <= 0, or
3. realized false-match rate among all auto-accepted test links > alpha = 0.05, or
4. realized missed-match rate among all auto-rejected test pairs > beta = 0.05, or
5. calibration yields no gray zone (T_L >= T_U): B2 alone already meets alpha and beta, there is nothing to triage.

Definitions: review load L = share of test pairs left in the gray zone. L_B2 uses B2's calibrated prior only; L_hyb uses the posterior after the LLM update; R = 1 - L_hyb / L_B2. The design's primary metric, the difference in percentage points, is L_hyb - L_B2 = -R x L_B2 and is reported alongside. The decision is stated on R because L_B2 is unknown before calibration.

**Secondary (pre-specified, not part of the kill test):** H2: R_LLM - R_FS > 0 with a 95% CI excluding 0, where R_FS is the same pipeline with a Fellegi–Sunter (Splink) decision in place of the LLM. Outcome labels: H1 passes and H2 holds = POSITIVE; H1 passes and H2 fails = MIXED (gain comes from having a second stage); H1 fails = KILLED.

**Stop Criterion:** Calibration sample = 3,000 anchor records from the calibration half. Test sample = anchors from the test half, taken in seeded random order, until the test gray zone (determined by B2 score and the frozen thresholds only, no labels) holds at least 600 pairs; the anchor that crosses 600 is included whole. No further data is drawn after that.

**Method:** B2 (`src/baseline.py::b2`) and the experiment 1 prompt (`src/score.py::prompt`) are frozen. B2 scores are calibrated by logistic regression on the calibration split; thresholds T_L/T_U are set there; gray-zone pairs get a binary LLM decision; sensitivity and specificity of that decision are estimated on calibration gray-zone pairs; posterior odds = prior odds x likelihood ratio; the same thresholds apply to the posterior; pairs still in the band go to clerical review.

**Output target:** article (methodological; experiment 1 + this experiment).
**Language:** en

---

## Frozen parameters

| Parameter | Value |
|---|---|
| alpha (max false-match rate among auto-accepted) | 0.05 |
| beta (max missed-match rate among auto-rejected) | 0.05 |
| KILLED margin | R >= 0.20 and CI lower bound > 0 |
| Calibration anchors | 3,000 |
| Test gray-zone target | >= 600 pairs |
| Calibration gray-zone LLM budget | all pairs, or a seeded random 800 if more |
| LLM | qwen2.5:14b via Ollama, local |
| LLM samples per pair, k | 5 |
| LLM sampling | temperature 0.7, top_p 1.0, seeds 1–5, num_predict 12 |
| LLM decision D | D = 1 if mean of parsed sample scores >= 0.5; unparsed samples dropped; all 5 unparsed -> no D, pair stays in gray zone |
| Sens/spec estimate | calibration gray zone, add-one (Laplace) smoothing |
| Fellegi–Sunter decision D_FS | Splink match_probability >= 0.5, same update rule |
| Bootstrap | 1,000 resamples over test anchors, seed 0 |
| Random seed for sampling and split | 0 |

Note on k-sampling vs. log-probabilities: the design allowed log-probabilities where available. The plan fixes k-sample majority for everything, because the frozen prompt returns a number, not yes/no, and Ollama log-probability support depends on the installed version. This choice is made here, before any data is seen.

## Data Sources

- ICE-ID raw release, Hugging Face `goldpotatoes/ice-id`, CC-BY 4.0 (de Carvalho et al. 2025, arXiv 2506.13792): `people.csv`, `parishes.csv`, `counties.csv`. Status: *acquire pending* (not on disk in this container).
- Experiment 1 pairs: `data/derived/pairs.jsonl` (repo root). Status: *on disk*, used only for exclusion.
- Candidate construction (from the design, made precise):
  - Records eligible: non-empty `person` identifier (about 50.2% of records are labelled; unlabelled records have unknown identity and are excluded on both sides), non-empty first name, patronym, sex, and integer birth year.
  - Waves: the 16 census waves in chronological order; anchor wave w_i, candidate wave w_{i+1}. Partial waves (1729, 1870) are kept and reported.
  - Candidates of an anchor: every eligible record in w_{i+1} with the same normalized first name, patronym and sex, and |birth year difference| <= 2. Label = same `person`.
  - Exclusion: any record whose (person, wave) appears in experiment 1's 200 pairs.
  - Split: anchor `person` identifiers are assigned to calibration or test 50/50 by seeded shuffle. Candidates may belong to persons in either half; resampling clusters on anchors.
- Literature for the article (not needed to run the experiment): Ather 2026; de Carvalho et al. 2025; Dasanaike 2026 (two papers); Peeters et al. 2025; Wang et al. 2025; OpenSanctions Pairs 2026. All indexed in `research/llm-vs-rules-er-brief.md` (repo root). Status: *acquire pending*; needs `.research-library` to be set.

## Tasks

### Task 1: Acquire and fingerprint ICE-ID
**Files:** `input/data/raw/` (gitignored), `input/data/README.md`, `.gitignore`

- [ ] Add `input/data/raw/` to `.gitignore`
- [ ] Download `people.csv`, `parishes.csv`, `counties.csv` from `goldpotatoes/ice-id` into `input/data/raw/`
- [ ] Record source URL, download date, row counts and SHA-256 of each file in `input/data/README.md`
- [ ] Check row count of `people.csv` = 984,028 and label rate approximately 50.2%; log any mismatch in `knowledge/_meta/log.md`

### Task 2: Build candidate pairs and split
**Files:** `output/data-analysis/make_candidates.py`, `input/data/derived/candidates.jsonl`, `input/data/derived/split.json`

- [ ] Write `make_candidates.py` implementing the eligibility, blocking, exclusion and split rules above (seed 0); records use the same field schema as experiment 1 pairs so `b2` runs unchanged
- [ ] Write `candidates.jsonl` (one line per pair: pair id, anchor id, split, a, b, label) and `split.json` (anchor person -> split)
- [ ] Print and save summary: anchors, pairs, positives, negatives per split and per wave pair; blocking recall (share of anchors whose true w_{i+1} record is among candidates)
- [ ] Unit check: no excluded (person, wave) from experiment 1 appears; no anchor person in both splits

### Task 3: Calibrate B2 and freeze thresholds (calibration split only)
**Files:** `output/data-analysis/calibrate_b2.py`, `output/data-analysis/results/thresholds.json`

- [ ] Draw the 3,000 calibration anchors (seeded order) and score their pairs with frozen `b2`
- [ ] Fit logistic regression label ~ b2 score; save coefficients
- [ ] Compute T_U (smallest calibrated p with false-match share <= alpha above it) and T_L (largest p with missed-match share <= beta below it); if T_L >= T_U, stop: criterion 5 applies, verdict KILLED, skip to Task 8
- [ ] Save `thresholds.json` (coefficients, T_L, T_U, calibration gray-zone size, L_B2 on calibration)
- [ ] Commit `thresholds.json` before any test label or LLM output is read

### Task 4: Draw the test sample and extract gray zones
**Files:** `output/data-analysis/draw_test.py`, `input/data/derived/gray_cal.jsonl`, `input/data/derived/gray_test.jsonl`

- [ ] Draw test anchors in seeded order until the test gray zone reaches >= 600 pairs (B2 score and frozen thresholds only; labels not read)
- [ ] Write `gray_cal.jsonl` (all calibration gray-zone pairs, or seeded 800) and `gray_test.jsonl` without the label field; keep labels in a separate `labels_test.jsonl`
- [ ] Commit the gray-zone files

### Task 5: Fellegi–Sunter second stage
**Files:** `output/data-analysis/fs_stage.py`, `output/data-analysis/results/fs_decisions.jsonl`

- [ ] Configure Splink comparisons on the B2 fields: birth year (exact / ±1 / ±2), parish, county, farm, partner (null-aware), distance in km (0 / <10 / <50 / else)
- [ ] Train u by random sampling and m by EM on calibration candidates only; save the model JSON
- [ ] Predict match_probability for all calibration and test gray-zone pairs; D_FS = probability >= 0.5; save `fs_decisions.jsonl`

### Task 6: LLM scoring of gray-zone pairs (researcher's machine)
**Files:** `output/data-analysis/score_gray.py`, `output/data-analysis/results/raw_qwen_gray.jsonl`

- [ ] Write `score_gray.py`: imports the frozen `prompt` from `src/score.py`, calls Ollama with the frozen sampling parameters, k = 5, resume-safe cache keyed by pair id and seed
- [ ] Researcher runs it locally on `gray_cal.jsonl` then `gray_test.jsonl`; record Ollama version and model digest in the output header
- [ ] Commit `raw_qwen_gray.jsonl` to the branch

### Task 7: Primary and secondary analysis
**Files:** `output/data-analysis/analyze.py`, `output/data-analysis/results/primary.json`

- [ ] Estimate LLM and FS sensitivity/specificity on calibration gray zone (Laplace smoothing)
- [ ] Apply the Bayesian update and frozen thresholds on the test split; compute L_B2, L_hyb (LLM), L_hyb (FS), R_LLM, R_FS, realized false-match and missed-match rates
- [ ] Cluster bootstrap (1,000, seed 0) for R_LLM, R_FS, R_LLM - R_FS
- [ ] Evaluate criteria 1–5 and H2 mechanically; write verdict (POSITIVE / MIXED / KILLED) to `primary.json`
- [ ] Report unparsed-sample rate and number of pairs with no D

### Task 8: Record results
**Files:** `output/data-analysis/RESULTS.md`, `knowledge/synthesis/gray-zone-triage-results.md`, `knowledge/_meta/log.md`

- [ ] Write `RESULTS.md`: frozen parameters, numbers table, verdict, deviations
- [ ] Write a synthesis page linking experiment 1 and this result; `status: review`
- [ ] Log the run and any deviations in `knowledge/_meta/log.md`

### Task 9: Literature for the article
**Files:** `input/bibliography/literaturguide.md`, `output/bibtex/references.bib`, `knowledge/sources/*.md`

- [ ] Set `.research-library` on the researcher's machine (blocking for ingest)
- [ ] Run `literature-review` (related work: LLM entity matching, hybrid rules + neural ER including the 2026 Ancestry paper "Do Rules Still Matter? Hybrid Entity Matching in the Neural Era", historical census linkage, the ICE-ID companion methods paper)
- [ ] Run `acquire-sources`, then `ingest-source` for the A-graded sources

### Task 10: Draft the article
**Files:** `output/article/article.qmd`, `output/article/outline/`

- [ ] Confirm `wiki-lint` green and synthesis pages stable
- [ ] Draft via `drafting-manuscript`: Introduction 1,200–1,800 words; Related work 1,500–2,500; Data and methods (both experiments) 2,500–3,500; Results 1,500–2,500; Discussion 1,500–2,500. Word counts are floors for development, not quotas.

## Verification

The project is complete when:
- [ ] `input/data/README.md` lists SHA-256 for all three raw files
- [ ] `thresholds.json` commit precedes the commits of `gray_test.jsonl` and `raw_qwen_gray.jsonl` in git history
- [ ] `primary.json` contains R_LLM, R_FS, both CIs, both realized error rates, and a verdict computed by `analyze.py` without manual editing
- [ ] H1 explicitly marked POSITIVE, MIXED or KILLED in `RESULTS.md`
- [ ] `scripts/lint-wiki.py` exits 0
- [ ] `make` in `output/article/` renders without errors
- [ ] Peer review round completed via `requesting-peer-review`

## Known gaps and deviation rules

- The ICE-ID companion methods paper (12 baselines including Ditto, XGBoost, Splink, per the v2 dataset paper) could not be retrieved before pre-registration: web search credits were exhausted and direct arXiv access is blocked in this environment. If Task 9 finds that it already reports an LLM gray-zone hybrid on ICE-ID, the result is logged and the article is reframed as a replication; H1 and its criteria do not change.
- Any change to frozen parameters after sign-off is logged in `knowledge/_meta/log.md` with the reason, and the affected result is labelled exploratory.

## Amendment 1 (2026-11-17, before any outcome data)

Approved by the researcher after the Task 2 reviews. At the time of the
amendment no threshold had been computed, no LLM call made, and no test-split
outcome inspected (only blocking counts). H1, the falsification criteria and
every value in "Frozen parameters" are unchanged. The text above is kept as
registered; where it conflicts with this section, this section applies.

1. **Leakage removed: `partner_id` is blanked in every input.** In ICE-ID the
   `partner` field holds the spouse's expert-curated `person` id (99.67% equal
   to the person of the `source_partner` row; filled for 13.2% of labelled and
   0.1% of unlabelled records). B2's cross-wave partner rule therefore reads
   the gold-standard linkage of the spouse. `make_candidates.py` writes
   `partner_id = ""` for every record, so B2 (code unchanged), the
   Fellegi–Sunter stage and the LLM prompt never see it. The Splink
   comparison on partner is dropped. `farm_id` is a place identifier (filled
   for 99.8% of unlabelled records) and stays. Experiment 1 check: 4 of 200
   pairs had a partner match; B2 AUC without it is 0.900 [0.856, 0.937] vs
   0.905, so KILLED-AT-4 is unaffected.
2. **Bootstrap unit made explicit.** "Resamples over test anchors" means
   resampling anchor persons (the split unit), with all their anchor records
   and pairs. This CI decides criterion 2.
3. **Added sensitivity analyses (reported, not decisive):**
   - CI clustered on name blocks (first name, patronym, sex, wave pair),
     because 1% of blocks hold about 72% of pairs;
   - primary metrics recomputed excluding anchors that touch a (person, wave)
     key with more than one record (likely merged homonyms in the labels);
   - anchor-weighted review load as a secondary figure.
4. **Added descriptive reporting:** end-to-end missed links (true links lost
   by blocking plus those auto-rejected), alongside beta; L and false-match
   rate per wave pair, descriptive only, no per-wave claims.
5. **Sex normalisation is case-insensitive** (`KArl` -> male), as "normalized
   sex" in Data Sources requires; one record affected.
6. **Storage format (no method change):** Task 2 writes `candidates.csv`
   (pair_id, anchor_rid, cand_rid, anchor_person, wave_a, wave_b, split,
   label) plus `records.jsonl` joined on rid, instead of one
   `candidates.jsonl` with inline records. The two large files are
   gitignored and verified by SHA-256 in `input/data/README.md`.
7. **Labels are taken as truth, uncorrected.** Candidate records may belong
   to persons of the other split; pairs are never filtered on the candidate's
   split, because positives always share the anchor's split and such a filter
   would depend on the label.

## Deviations Log

Append to `knowledge/_meta/log.md` whenever scope, method, or data sources change after pre-registration. Reference the slug `gray-zone-triage`.
