---
title: "A hand-built rule beats LLMs at linking Icelandic census homonyms: a preregistered negative result"
date: 2026-11-17
bibliography: references.bib
status: draft
---

## Abstract

Historical record linkage decides whether two census entries describe the same
person. Under patronymic naming, which Galton singled out as the hard case of
identification [@galton1892finger], the name alone almost never decides it. We
ask whether a large language model, reading the structured fields of two
records, links them better than a deterministic rule. On 200 preregistered
pairs from the ICE-ID Icelandic census benchmark [@carvalho2025iceid] (100
same-person pairs across census waves, 100 hard negatives: different people
with the same name, sex and birth year ±1), a hand-weighted rule scores ROC-AUC
0.905 [95% CI 0.862, 0.940]. claude-sonnet-5 scores 0.816 [0.758, 0.869] and
qwen2.5:14b 0.787 [0.728, 0.843]. The rule is better than the commercial model
by 0.089 [0.045, 0.137], so the preregistered criterion (beat the rule by at
least 0.05) fails. Post-hoc we found and removed four design leaks in the 200
pairs, built held-out sets of 4,000-5,000 pairs drawn at random from the full
blocked candidate space, and added the canonical probabilistic baseline. With
its weights frozen, the rule (0.961) matches Fellegi-Sunter fitted with Splink
(0.959) and a dev-fitted logistic regression (0.963). Stacking the LLM score on
top of the rule does not improve it. On clean structured fields, the LLM adds
nothing a rule does not already capture.

## 1. Introduction

Linking people across censuses underpins historical demography and economics
[@ruggles2018historical; @abramitzky2021automated]. Since @newcombe1959automatic
and @fellegi1969theory the standard tool has been a probabilistic model that
weighs agreement on each field by how much more often it occurs among matches
than among non-matches; Splink is a widely used modern implementation
[@linacre2022splink]. Automated linkers are fast but make systematic errors
[@bailey2020how], and hand linkage by genealogists remains the reference.

LLMs are now strong entity matchers on product and bibliographic benchmarks
[@narayan2022can; @peeters2025entity], ahead of earlier deep matchers
[@mudgal2018deep; @li2020deep]. Those benchmarks are largely solvable from the
names of the entities [@papadakis2024critical]. Icelandic census data removes
that shortcut: names are patronymic and repeat constantly, so the decision
rests on birth year, place, household and life course. If an LLM's "reasoning
over messy evidence" adds value anywhere in linkage, it should show where the
name is useless.

We test this with a preregistered kill criterion and report a clean negative.
Our contribution is the negative result and the checks that make it hold: a
canonical Fellegi-Sunter baseline, held-out sets twenty times larger, an audit
of the original pair set, and a test of the LLM as an extra feature rather than
a replacement.

## 2. Related work

**Probabilistic linkage.** @fellegi1969theory formalised linkage as a
likelihood-ratio test under conditional independence of fields; parameters can
be estimated without labels by EM. @christen2012data covers blocking,
comparison and classification. Splink [@linacre2022splink] implements this
model at scale and is our canonical baseline.

**Historical census linkage.** @ruggles2018historical review census linkage and
its uses; @abramitzky2021automated compare automated methods with hand linkage
in US censuses, and @bailey2020how show that automated methods can produce
many false links that bias downstream estimates. ICE-ID
[@carvalho2025iceid] releases 984,028 Icelandic census records (1703-1920) with
226,864 expert-curated person identifiers. Its v1 preprint reports pairwise
AUC 0.999 for a gradient-boosting ensemble across waves, with negatives that
need not share a name; those numbers measure an easier task and are not
comparable to ours (Section 5).

**LLMs for entity matching.** @narayan2022can showed that foundation models
match entities with few or no labelled examples; @peeters2025entity compare
prompting strategies and fine-tuning across LLMs. Both evaluate on benchmarks
where textual identity dominates.

**Contamination.** Public benchmarks may sit in model training data
[@sainz2023nlp; @golchin2024time]. ICE-ID has been public since 2025; we
discuss this in Section 6.

## 3. Data and pair design

**v1 (preregistered).** 100 positives: for a person seen in at least two waves,
the two appearances with the smallest wave gap. 100 hard negatives: two
different person ids with the same lowercased first name, patronym and sex,
birth years within 1, from different waves. Both models and baselines saw name,
sex, birth year, census year, household status, marital status, parish and
county (with coordinates), farm id and partner id.

**Problems found after the run.** (i) The census gap alone separates the
classes (AUC 0.831), because negatives come from any two waves. (ii) 16
negatives, and no positive, have record A in the later wave. (iii) 20
negatives, against 1 positive, differ in full name (middle names). (iv) The
`partner` field is the partner's expert person id, so equal values across waves
reflect the gold linkage; it affects 4 positives. Selection also took the first
N items in sort order, so all negatives have first names beginning with "A".
(i) and (ii) were visible to the LLMs but not used by the rule; (iv) helped the
rule by 0.005 AUC.

**v2 (post-hoc).** The candidate space is every pair of labelled records in two
waves with identical normalized full name, same sex and birth years within 1:
the output of a standard blocking step. Positives are pairs with the same
person id. Balanced sets pair each sampled positive with a negative from the
same wave pair; a natural set samples the adjacent-wave block uniformly
(prevalence 0.257). Partner is shown by name via `source_partner`, a row
reference within the same wave. Dev and test are split by a hash of the name.
Sets: dev_adjacent and test_adjacent (4,000 each), test_longgap (4,000, waves
≥ 20 years apart), test_natural (5,000), and llm_adjacent (400, a wave-matched
subset of test_adjacent).

## 4. Methods

**Baselines.** B1: birth-year agreement. B2: a hand-set additive rule (birth
year 2/1, parish 1.5, county 0.5, farm 1, partner 2, distance decaying to 0 at
100 km); its v1 weights are frozen for v2. **Fellegi-Sunter** on seven
comparison fields (birth year, parish, county, farm, partner, distance at
1/10/50 km, and a life-course check that flags married or widowed followed by
never-married): m/u by unsupervised EM (FS_em) or by counting dev labels
(FS_sup), implemented in numpy; and **Splink 5.0** with the same fields, u by
random sampling over 221,839 dev records and m by EM over same-name blocks,
without labels. **LR**: L2 logistic regression on one-hot comparison levels,
fitted on dev.

**LLMs (v1).** A fixed zero-shot prompt asks for a probability from 0 to 1 that
the two records are the same person, explaining that names are uninformative.
qwen2.5:14b ran locally via Ollama at temperature 0. claude-sonnet-5 ran once
per pair through the Claude CLI, which has no temperature setting and which
added our system instruction to its own default system prompt.

**Evaluation.** ROC-AUC with 1,000-resample bootstrap CIs; paired bootstrap
(2,000 resamples) for differences; average precision for the natural set.
**Kill criterion (v1, preregistered):** the commercial model needs AUC ≥ 0.65,
a CI excluding 0.5, and a gain of at least +0.05 over the best baseline.

**LLM as feature.** On v1, cross-fitted logistic stacking (10 folds, 20
repeats) of the rule score and the LLM's logit, scored by the fitted slope
direction so that fold intercepts cannot reshuffle the pooled ranking.

## 5. Results

**v1.** Rule 0.905, claude-sonnet-5 0.816, qwen2.5:14b 0.787, birth year alone
0.722. Commercial minus rule: −0.089 [−0.137, −0.045]. Commercial minus local:
+0.030 [−0.037, +0.095]. Verdict: killed. Without the gold-derived partner
term the rule still leads by 0.084 [0.040, 0.132]. Census gap alone (0.831)
beats both LLMs, although they could see it.

**v2 baselines** (AUC; Δ vs frozen B2 with 95% CI):

| method | test_adjacent | test_longgap | test_natural (AP) |
|---|---|---|---|
| B2, frozen | 0.961 | 0.890 | 0.968 (0.920) |
| FS_em | 0.958 (−0.003 [−0.007, +0.001]) | 0.895 (+0.005 [−0.001, +0.010]) | 0.966 (0.921) |
| FS Splink | 0.959 (−0.002 [−0.005, +0.002]) | 0.896 (+0.006 [+0.001, +0.012]) | 0.967 (0.922) |
| LR, dev-fitted | 0.963 (+0.003 [−0.000, +0.006]) | 0.900 (+0.010 [+0.005, +0.015]) | 0.972 (0.935) |

The hand-set rule did not depend on having seen the v1 pairs: with frozen
weights it matches fitted models on fresh adjacent-wave pairs and trails them
by at most 0.010 on long gaps. Long-gap linkage is harder for every method.

**LLM as feature (v1).** Rule alone 0.900; with claude-sonnet-5 0.894
(Δ −0.006 [−0.017, +0.004]); with qwen2.5:14b 0.897 (Δ −0.003 [−0.016, +0.010]).

**Comparison with ICE-ID's reported results.** ICE-ID v1 reports AUC 0.999 for
across-wave linkage with supervised ensembles trained on up to 500,000 pairs
whose negatives are different-person pairs that need not share a name. Our sets block on the identical name, so
the name carries no signal for either class. The two numbers describe different
tasks; together they suggest the name-distinct task is close to saturated and
the homonym task is where models are still separated.

## 6. Discussion

The fields that decide homonym links are categorical and numeric. A weighted
sum over them, whether hand-set or estimated by Fellegi-Sunter, already
captures what the LLM extracts from the same fields written as text, and the
LLM's score adds nothing once the rule's is known. The commercial model's
advantage over a 14B local model is within noise.

**Contamination.** ICE-ID, including the person ids, is public, and we do not
know either model's training data. Contamination could only raise the LLM
scores, so it cannot explain a negative result. A direct probe (ask for the row
ids of the same person in other waves and check them) is left for future work.

## 7. Limitations

- LLM results exist only for the 200 v1 pairs, which carry the design problems
  above. The v2 sets are built and the scoring code supports few-shot prompts,
  repeated samples and a temperature-0 API backend, but no LLM has been run on
  v2.
- One prompt and one zero-shot format were tested on v1. Few-shot prompting and
  fine-tuning are untested.
- The commercial scores are single samples from the CLI, without temperature
  control, and the CLI's default system prompt was in effect. Model snapshot,
  CLI version, Ollama version and model digest were not logged for v1.
- Only clean structured fields were tested. Free-text, OCR-damaged or
  inconsistently transcribed records, where an LLM might help, are not.
- The labels are expert curations and may miss links. Unlabelled rows (half
  the dataset) are excluded, so the "natural" prevalence is that of the
  labelled subset.
- One country and one benchmark.

## 8. Reproducibility

The code and cached model outputs are in this repository (code MIT; derived
data CC-BY 4.0). `people.csv` SHA-256 `81e30e5c…5d49d1`. Every number above is
recomputed by the scripts in `README.md`.

## References
