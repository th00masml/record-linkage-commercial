# Gray-zone LLM triage on ICE-ID: results

Run date: 2026-11-17. Plan: `input/ideas/gray-zone-triage-plan.md`
(pre-registered, Amendment 1 before any outcome data). All numbers come from
`results/primary.json`, written by `analyze.py` without manual editing.

## Verdict: KILLED

The hybrid (B2 rule + qwen2.5:14b on B2's gray zone, Bayesian update) removes
4.7% of B2's clerical-review load, 95% CI [3.0%, 6.6%]. The pre-registered
minimum was 20%. Falsification criterion 1 is triggered; criteria 2–5 are not.

| Criterion | Registered limit | Observed | Triggered |
|---|---|---|---|
| 1. R | >= 0.20 | 0.047 | yes |
| 2. CI lower bound of R | > 0 | 0.030 | no |
| 3. False-match rate, auto-accepted | <= 0.05 | 0.033 | no |
| 4. Missed-match rate, auto-rejected | <= 0.05 | 0.049 | no |
| 5. Gray zone exists | T_L < T_U | yes | no |

## Numbers

Test sample: 22,258 candidate pairs from 5,112 anchor records (4,895 anchor
persons), sealed until the LLM output was committed. Commit order verified:
thresholds `6b0e803` -> test gray zone `c33e5e0` -> LLM output `ac1134d`.

| Method | Review load L | R | 95% CI (anchor persons) | False-match | Missed-match |
|---|---|---|---|---|---|
| B2 alone | 2.70% (600 pairs) | 0 | | 0.033 | 0.048 |
| B2 + qwen2.5:14b | 2.57% (572 pairs) | 0.047 | [0.030, 0.066] | 0.033 | 0.049 |
| B2 + Fellegi–Sunter (registered, D at 0.5) | 2.70% (600 pairs) | 0 | [0, 0] | 0.033 | 0.048 |

The LLM resolved 28 of 600 gray pairs (2 to accept, 26 to reject).

## Why so little

Second-stage accuracy on the 329 calibration gray pairs (250 matches, 79 non-matches):

| | Says "same" | Sensitivity | Specificity | LR if "same" | LR if "different" |
|---|---|---|---|---|---|
| qwen2.5:14b (k = 5, mean >= 0.5) | 96% | 0.976 [0.954, 0.991] | 0.111 [0.052, 0.187] | 1.10 | 0.21 |
| Fellegi–Sunter (registered) | 100% | 1.0 | 0.0 | 1.01 | 0.32 |

Inside the band the LLM says "same person" for 96% of pairs. A "same" verdict
barely moves the prior (LR 1.10), so it cannot push a pair over T_U. A
"different" verdict is informative (LR 0.21) but comes for about 4% of pairs.
The LLM identified 8 of the 79 non-matches. The band is exactly where B2's
structured evidence is ambiguous, and the LLM reads the same fields.

## H2 (secondary) and the exploratory comparator

The registered H2 (R_LLM - R_FS > 0) formally holds, CI [0.030, 0.066], but
only because the registered Fellegi–Sunter decision is constant in the band
(logged at Task 5). It carries no information and should not be cited as
"the LLM beats Fellegi–Sunter".

Exploratory, not part of the verdict: with the Fellegi–Sunter threshold chosen
on the calibration gray zone (maximal Youden J, threshold p_FS = 1.0), the
non-LLM second stage reaches R = 0.090 [0.066, 0.111], higher than the LLM
(difference LLM - FS: [-0.071, -0.011]). It also breaches beta (missed-match
0.0507), so it would be KILLED under the same criteria.

## Sensitivity (Amendment 1, not decisive)

- CI clustered on name blocks (4,353 blocks): R in [0.029, 0.067], same as by anchor persons.
- Anchor-weighted review load: R = 0.034.
- Without anchors touching a duplicate (person, wave) key: R = 0.047, but this
  drops 14,150 of 22,258 pairs (the flag catches most common-name blocks) and
  the remaining set's missed-match rate is 0.132. The duplicate-key filter is
  too coarse to be informative here; reported for completeness.

## Descriptive

- Recall of auto-accepted links: 2,747 of 4,125 true links in the sample (66.6%);
  924 true links (22%) are auto-rejected at beta = 0.05, 454 stay in review.
  Blocking itself loses 4,035 true links across all waves (recall 0.982).
- Review load by wave pair ranges from 1.3% (1703–1729) to 7.6% (1801–1816,
  1890–1901); the LLM changes it by at most 0.4 points in any wave pair.
- LLM output: 4,645 samples, all parsed; Ollama 0.40.2, model digest
  7cdf5a0187d5, temperature 0.7, seeds 1–5.

## Limits

- One LLM (qwen2.5:14b, local), one prompt (frozen from experiment 1), k = 5.
  A different prompt or model was not tested and is out of scope.
- Labels are the expert identifiers, taken as truth; label noise attenuates
  every difference.
- Thresholds were set on the calibration sample; the realized missed-match
  rate on test (0.049) sits close to beta, as anticipated in the Task 3 review.
- Amendment 1 removed `partner_id` (gold-standard leakage) from all inputs
  before any outcome was observed.
