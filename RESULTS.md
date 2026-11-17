# Results

Run date (v1, preregistered): 2026-08-09. Instruments: `qwen2.5:14b` (local)
and `claude-sonnet-5` (commercial, via the authenticated Claude CLI, no API
key). Numbers recompute from cached outputs in `outputs/`.
Post-hoc analysis, v2 baselines and the v2 / bare-instrument Sonnet runs:
2026-11-17 (sections marked post-hoc).

## Verdict: KILLED-AT-4

No LLM beats a hand-built deterministic linker by the preregistered margin
on structured Icelandic census linkage, on the preregistered set or on any
post-hoc set. On the preregistered 200 pairs the rule is significantly better
than both LLMs. Post-hoc, two of the original conclusions need qualifying:

- On the leak-free v2 set, claude-sonnet-5 called without Claude Code's own
  context **ties** the rule (0.972 vs 0.969, Δ +0.003 [−0.009, +0.015]). It is
  as good as the rule, not worse; it is still nowhere near +0.05 better.
- About half of the v1 gap was the instrument: the same model, called bare,
  scores 0.856 on v1 instead of 0.816 (+0.040 [−0.004, +0.087]). The rule still
  beats it there (−0.049 [−0.079, −0.020]).

The practical conclusion stands: a free, transparent rule (or Splink) matches the
frontier model on clean structured fields, so the LLM call buys nothing here.

## The numbers (v1, preregistered)

200 pairs: 100 same-person-across-adjacent-waves positives + 100 homonym
hard negatives (same first name + patronym + sex, born within 1 year,
different people).

| method | AUC | 95% CI |
|---|---|---|
| B1 birth-year only | 0.722 | [0.659, 0.787] |
| **B2 hand-built rule** | **0.905** | [0.862, 0.940] |
| M_local qwen2.5:14b | 0.787 | [0.728, 0.843] |
| M_comm claude-sonnet-5 | 0.816 | [0.758, 0.869] |

Primary (commercial vs best baseline):
- gain over B2 = **−0.089** (needed +0.05)
- paired ΔAUC (M_comm − B2) = **−0.089**, 95% CI [−0.137, −0.045]; the share
  of bootstrap deltas ≤ 0 is 1.000. The rule is significantly better.
- VERDICT: **KILLED-AT-4**.

```
cd src
python prepare_pairs.py
python baseline.py            # B1, B2 (run first)
python score.py local && python score.py comm
python report.py
```

### Commercial vs local: not worth it

paired ΔAUC (M_comm − M_local) = +0.030, 95% CI [−0.037, +0.095], share of
bootstrap deltas ≤ 0 = 0.20. The frontier model is not significantly better
than the local 14B at this task.

## Problems found in the v1 pair set (post-hoc)

`python src/analysis_v1.py` → `outputs/analysis_v1.json`.

| problem | evidence on v1 |
|---|---|
| census gap separates the classes (positives = adjacent appearances, negatives = any two waves) | gap alone: AUC **0.831**, above both LLMs |
| record order leaks | 16 negatives have A in the later wave, 0 positives |
| full name not identical in negatives | 20 negatives vs 1 positive differ (middle names); name-identity alone: AUC 0.595 |
| non-random selection ("first N in sort order") | all 100 negatives have first names starting with "A"; positives are the lowest person ids as strings |
| `partner_id` is the partner's expert person id (gold-derived) | 4 positives use it; B2 without it: 0.900 (vs 0.905) |

The LLMs saw census years, so the first two problems, if anything, favoured
them over B2 (which ignores census years). B2 without the gold-derived partner
term still beats claude-sonnet-5 by 0.084, 95% CI [−0.132, −0.040]. The v1
verdict stands.

### The v1 commercial instrument (post-hoc)

The v1 call used `claude -p --append-system-prompt`, which adds our instruction
to Claude Code's own default system prompt, tool definitions and installed
skills (~65k tokens per call in this environment). Re-run on the same 200 pairs
with that context stripped (`--system-mode replace`: our system prompt and the
two records only, ~1.4k tokens), 3 samples per pair, CLI 2.1.326, model id
reported per call `claude-sonnet-5`
(`outputs/pairs__comm__cli__shots0__sysreplace/`):

| | AUC [95% CI] | Δ vs v1 call | Δ vs B2 |
|---|---|---|---|
| v1 call (Claude Code context, 1 sample) | 0.816 [0.758, 0.869] | — | −0.089 [−0.137, −0.045] |
| bare, mean of 3 samples | 0.856 [0.801, 0.905] | +0.040 [−0.004, +0.087] | −0.049 [−0.079, −0.020] |
| bare, first sample only | 0.860 | +0.044 [+0.000, +0.091] | |

Per-sample AUC 0.860 / 0.851 / 0.851; mean within-pair SD of the score 0.035.

## v2: larger, leak-controlled, held-out sets (post-hoc)

`prepare_pairs_v2.py` → `baselines_v2.py`, `splink_baseline.py` → `report_v2.py`.

Design. The candidate space is what a real linker faces after blocking: every
pair of labelled records in two census waves with the **identical full name**,
same sex and birth years within 1. A pair is positive iff the expert person ids
match. Names therefore carry no signal for either class. Negatives are drawn
from the **same wave pair** as a positive, so census years carry none either;
record A is always the earlier wave; sampling is random (seed 0); the partner is
shown by name (looked up through `source_partner`, a row id within the same
wave), not by gold id. Dev and test are split by a hash of the name, so no name
or person is in both; every fitted weight comes from dev only.

| set | pairs | prevalence | waves |
|---|---|---|---|
| dev_adjacent | 4,000 | 0.50 | consecutive waves 1801-1920 |
| test_adjacent | 4,000 | 0.50 | consecutive waves |
| test_longgap | 4,000 | 0.50 | wave pairs ≥ 20 years apart |
| test_natural | 5,000 | 0.257 | uniform draw from the adjacent-wave block (realistic class mix) |
| llm_adjacent | 400 | 0.50 | wave-matched subset of test_adjacent, for LLM runs |

The full adjacent-wave block has 221,790 positive and 530,110 negative pairs
(prevalence 0.295); the long-gap block 224,665 / 675,043 (0.250).

Methods: B1, B2 with its **v1 weights frozen** (partner term on names),
Fellegi-Sunter with m/u by unsupervised EM (FS_em) and by counting dev labels
(FS_sup), **Splink 5.0** (u by random sampling over 221,839 dev-split records,
m by EM over same-name blocks; no labels), and L2 logistic regression on one-hot
comparison levels (LR). Comparison fields for FS/Splink/LR: birth year, parish,
county, farm, partner name, distance (≤1/10/50 km), and a life-course check
(married or widowed, later never-married).

ROC-AUC [95% bootstrap CI]; AP = average precision; Δ = paired ΔAUC vs B2.

| method | test_adjacent | Δ vs B2 | test_longgap | Δ vs B2 | test_natural (AP) |
|---|---|---|---|---|---|
| B1 | 0.689 [0.674, 0.703] | −0.272 | 0.620 [0.606, 0.636] | −0.270 | 0.684 (0.244) |
| **B2 (frozen v1 weights)** | **0.961** [0.955, 0.966] | — | 0.890 [0.881, 0.899] | — | 0.968 (0.920) |
| FS_em | 0.958 [0.951, 0.964] | −0.003 [−0.007, +0.001] | 0.895 [0.885, 0.905] | +0.005 [−0.001, +0.010] | 0.966 (0.921) |
| FS_sup | 0.960 [0.953, 0.965] | −0.001 [−0.004, +0.002] | 0.897 [0.887, 0.907] | +0.007 [+0.001, +0.012] | 0.967 (0.923) |
| FS_splink | 0.959 [0.952, 0.965] | −0.002 [−0.005, +0.002] | 0.896 [0.886, 0.906] | +0.006 [+0.001, +0.012] | 0.967 (0.922) |
| LR (dev-fitted) | 0.963 [0.958, 0.968] | +0.003 [−0.000, +0.006] | 0.900 [0.890, 0.909] | +0.010 [+0.005, +0.015] | 0.972 (0.935) |

Reading:
- **The hand-set rule is not overfitted.** With its weights frozen, B2 on fresh
  random adjacent-wave pairs is within 0.003 AUC of every model fitted on a
  separate dev split, including Splink. On adjacent waves no fitted model beats it significantly.
- **Fellegi-Sunter and Splink agree with each other** (0.958-0.960) and with B2.
  The canonical method is not stronger than the rule here; the rule is not a
  straw man.
- **Long-gap links are harder** (0.89-0.90 for every method). Fitted models gain
  0.005-0.010 over B2 there, a small but significant amount.
- **Class balance.** At the realistic 25.7% prevalence the ranking quality is
  unchanged (AUC 0.966-0.972) and AP is 0.92-0.94 (B1: 0.24, the prevalence
  floor is 0.257).
- v1 vs v2: B2 scores 0.905 on v1 and 0.961 on v2 test_adjacent. v1 is the harder
  set because its negatives were chosen to be maximally confusable (first
  matching pair per name key), and its positives include longer gaps.

### LLMs on v2 (`llm_adjacent`, 400 pairs)

claude-sonnet-5 via the CLI with `--system-mode replace` (bare instrument),
zero-shot, 3 samples per pair, scores averaged; CLI 2.1.326; the model id the
CLI reported on every call was `claude-sonnet-5`
(`outputs/v2_llm_adjacent__comm__cli__shots0__sysreplace/`).

| method | AUC [95% CI] | Δ vs B2 [95% CI] |
|---|---|---|
| B2 (frozen) | 0.969 [0.951, 0.984] | — |
| FS_splink | 0.970 [0.953, 0.985] | +0.001 [−0.009, +0.011] |
| LR (dev-fitted) | 0.972 [0.955, 0.986] | +0.003 [−0.004, +0.010] |
| claude-sonnet-5, bare, mean of 3 | 0.972 [0.955, 0.986] | +0.003 [−0.009, +0.015] |

Per-sample AUC 0.971 / 0.970 / 0.971; mean within-pair SD of the score 0.022.
Gain over the best baseline is +0.003 against the +0.05 the preregistered
criterion asks for: the criterion fails on v2 too, but here because the LLM
ties the rule, not because it loses to it. Together with the v1 re-run this
says the v1 gap came from two sources: the instrument (about 0.04) and the v1
pair set (the remaining 0.05). Which property of the v1 set hurts the LLM more
than the rule (the first-match-per-name negatives, the narrow "A" name sample,
the mixed census gaps) is not isolated here.

**qwen2.5:14b on v2: not run.** The cloud environment this was run in has no
GPU and cannot reach the Ollama registry. Commands for a machine with Ollama:

```
cd src
python score.py local --pairs v2/llm_adjacent
python score.py local --pairs v2/llm_adjacent --shots 4
python report_v2.py && python stacking_v2.py
```

## LLM as a feature, not a replacement (post-hoc, v1)

Does the LLM add anything **on top of** the rule? Cross-fitted logistic
stacking on v1 (10-fold, 20 repeats, slope direction only so fold intercepts
cannot reshuffle the pooled ranking), B2 without the gold partner term as the
base:

| model | AUC | Δ vs B2 alone [95% CI] |
|---|---|---|
| B2 (no partner id) | 0.900 | — |
| B2 + logit(claude-sonnet-5) | 0.894 | −0.006 [−0.017, +0.004] |
| B2 + logit(qwen2.5:14b) | 0.897 | −0.003 [−0.016, +0.010] |

Neither LLM score adds information the rule lacks. (A first version of this
analysis averaged raw out-of-fold log-odds and showed a spurious +0.017; the
fold intercepts had depressed the B2-only baseline to 0.867. Fixed.)

On v2 `llm_adjacent` (`python src/stacking_v2.py`), B2 + logit(claude-sonnet-5,
bare) scores 0.975 against B2's 0.969: Δ +0.006 [−0.002, +0.015], share of
bootstrap deltas ≤ 0 = 0.07. A small gain that the 400 pairs cannot
distinguish from zero.

Few-shot prompting (`--shots k`, examples from v2 dev, names disjoint from test)
is implemented; it has not been run for either model (see the qwen commands above).

## Why the LLM does not beat the rule

The disambiguating evidence here is structured: birth-year agreement,
parish/county, farm, partner, and geographic distance from coordinates.
A weighted rule over those fields is exactly the right tool, and on 4,000
fresh pairs it is as good as a Fellegi-Sunter model fitted with Splink. The
frontier model, reading the same fields as prose, reaches the same ceiling on
clean pairs (v2) and falls short of it on the hardest homonyms (v1). Stacked
on the rule it adds at most a few thousandths of AUC.

Where an LLM might still help is the genuinely messy tail (OCR-garbled
names, inconsistent free-text, missing fields): not tested here, and the
honest next step. On clean structured fields, use a rule.

## Memorization

Unlikely to matter, but not excluded. The models are never shown the person
ids, the names are common Icelandic patronymics shared by thousands, and
recalling which of 984,028 rows share an expert id would require memorizing a
large table nearly verbatim. Two facts keep this from being "by construction":
ICE-ID, including the `person` column, has been public on HuggingFace since
mid-2025, and we do not know the training cutoff or data of either model.
Contamination would inflate the LLM scores, so it cannot explain why the LLM
fails to beat the rule; at most it could mean the v2 tie overstates the LLM. A direct probe (show a row, ask for
the row ids of the same person in other waves; check against `people.csv`) is
straightforward and left as future work.

## Limits

- v1: 200 pairs, adjacent-appearance positives, the design problems above.
- v2 covers 4,000-5,000 pairs per test set, long gaps and the realistic class
  mix for the baselines. LLM results on v2 cover claude-sonnet-5 on the 400-pair
  adjacent-wave subset only (zero-shot, 3 samples); qwen2.5:14b on v2, few-shot
  prompting, long-gap and natural-prevalence LLM runs are still missing.
- One country, clean structured fields (the LLM's worst case relative to rules;
  its best case, messy free text, is untested).
- Labels are expert curations; two different person ids can be the same human
  (curation misses). Records without a person id (50% of rows) are outside both
  versions, so the v2 "natural" prevalence is that of the labelled subset.
- The v1 commercial scores are single CLI samples with no temperature control,
  run through Claude Code's default system prompt (`--append-system-prompt`).
  The post-hoc runs use 3 samples per pair (spread above) but still no
  temperature control; `--backend api` (temperature 0) has not been run.
- Ollama version, model digest, CLI version and the dated model snapshot behind
  `claude-sonnet-5` were not logged in v1 (see DATA-LICENSES.md); new runs log them.
