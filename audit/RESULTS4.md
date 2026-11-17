# Results 4: Splink on the within-wave task, and the kin-ID leakage check

Protocol: `PROTOCOL4.md` (committed before scoring). Code:
`splink_within_wave.py` (reuses `noise_effect.py`). Numbers:
`noise_effect_results_p4.json`, `splink_p4_extra.json`; trained model
`splink_within_wave_model.json`. Splink 4.0.17, DuckDB 1.x/2.0, trained on
493,681 labelled rows with no labels and no curated IDs.

## AUC, primary (audited sample, n = 437) and secondary (rule-cleaned, n = 30,700)

| method | ICE-ID | audited | change [95% CI] | ICE-ID (all) | rule-cleaned | change |
|---|---|---|---|---|---|---|
| B2 / B2_noid | 0.948 | 0.949 | +0.001 [−0.040, +0.039] | 0.939 | 0.922 | −0.017 |
| LR_noid | 0.975 | 0.941 | −0.034 [−0.080, +0.003] | 0.979 | 0.944 | −0.035 |
| GB_noid | 0.979 | 0.942 | −0.037 [−0.087, +0.001] | 0.984 | 0.944 | −0.040 |
| Splink | 0.917 | 0.929 | +0.011 [−0.035, +0.054] | 0.908 | 0.901 | −0.007 |

## E5, leakage (holds)

Dropping the curated partner/father/mother ID features changes nothing that
matters. B2 is identical (the partner term never fires on these pairs). The
learners' lead over B2_noid: GB +0.031 [+0.010, +0.057] under ICE-ID labels,
−0.007 [−0.034, +0.016] under audited labels; LR +0.027 → −0.008. The
protocol 3 finding does not rest on the leak that PR #1 found.

## E3, Splink loses less (holds, directionally)

Splink's AUC does not fall under audited labels (+0.011), while the learners
lose about 0.035; on all pairs it loses 0.007 against 0.035-0.040. Splink's
deficit to GB shrinks from −0.062 [−0.088, −0.036] (ICE-ID) to −0.013
[−0.038, +0.016] (audited). A model that never saw the labels is not hurt by
their noise. It is still the weakest discriminator of the label-free methods:
below B2 under both conditions (−0.031 [−0.057, −0.005] ICE-ID,
−0.020 [−0.038, −0.007] audited).

## E4, what Splink links (holds, but see below)

| threshold | audited two-human pairs linked | true duplicates linked |
|---|---|---|
| Splink's own, match probability ≥ 0.5 | 0.05 [0.01, 0.10] | 0.23 |
| F1-max under ICE-ID labels | 0.99 [0.96, 1.00] | 1.00 |
| (GB_noid, F1-max under ICE-ID) | 0.92 [0.85, 0.97] | 1.00 |

At its own threshold Splink almost never links two different people, but it
misses 77% of real duplicates: the threshold is conservative, not sharp.
Tuned on ICE-ID labels, the same model links 99% of the two-human pairs.

## Reading

The damage does not depend on the model class. A threshold or a model fitted
to ICE-ID's within-wave labels learns to link different people, whether the
model is boosting, logistic regression or Fellegi-Sunter. Splink is protected
only while it keeps its own probabilities, and then its recall is poor.

## Limits

As in RESULTS3: 19 true positives in the audited condition, negatives not
audited, one auditor. The Splink prior comes from a deterministic rule and
moves the 0.5 threshold; EM could not train three distance levels in the
second session (blocked on parish), they come from the first.
