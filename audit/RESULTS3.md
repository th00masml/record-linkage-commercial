# Results 3: what the label noise does to method results (within-wave task)

Protocol: `PROTOCOL3.md` (committed before scoring). Code: `noise_effect.py`
(pre-registered), `noise_effect_extra.py` (not pre-registered). Numbers:
`noise_effect_results.json`, `noise_effect_extra.json`.

Evaluation set: 10,288 within-wave positives, 20,412 negatives (two per
positive, same wave and name block; 61 positives had no negative). Primary
condition: the 150 audited pairs minus 13 undetermined = 137 audited positives,
of which 118 are two humans and 19 are true duplicates, plus 300 negatives;
weighted by stratum.

## Primary: audited labels (pre-registered)

| method | AUC ICE-ID | AUC audited | change [95% CI] | F1 ICE-ID | F1 audited |
|---|---|---|---|---|---|
| name_age | 0.835 | 0.872 | +0.037 [-0.075, +0.130] | 0.797 | 0.347 |
| B2 | 0.948 | 0.949 | +0.001 [-0.040, +0.039] | 0.863 | 0.297 |
| LR_noisy | 0.976 | 0.942 | -0.034 [-0.079, +0.005] | 0.900 | 0.293 |
| GB_noisy | 0.980 | 0.945 | -0.036 [-0.082, +0.002] | 0.924 | 0.306 |

Ranking under ICE-ID labels: GB, LR, B2, name_age. Under audited labels:
B2, GB, LR, name_age, but the top three are within 0.007 AUC of each other.

E1 (name_age and learners lose, B2 gains): partly. The learners lose about
0.035 AUC with intervals that just reach zero; B2 is flat; name_age does not
lose. E2 (ranking changes): the nominal top position changes, but not beyond
the uncertainty.

## Not pre-registered: the learners' lead over B2, paired

| | ICE-ID labels | audited labels |
|---|---|---|
| GB_noisy − B2 | +0.032 [+0.011, +0.057] | −0.004 [−0.029, +0.017] |
| LR_noisy − B2 | +0.028 [+0.005, +0.053] | −0.007 [−0.029, +0.012] |

Against ICE-ID labels, models trained on those labels beat the hand rule
significantly. Against audited labels the lead is gone. On all pairs with
rule-cleaned labels (secondary, partly circular) it halves: GB +0.045 to
+0.023, LR +0.041 to +0.022, both intervals excluding zero.

## Not pre-registered: what the methods link

At the threshold that maximises F1 under ICE-ID labels, share of the audited
two-human pairs each method links:

| method | two-human pairs linked [95% CI] | true duplicates linked |
|---|---|---|
| name_age | 0.67 [0.57, 0.77] | 0.89 |
| B2 | 0.87 [0.80, 0.94] | 1.00 |
| LR_noisy | 0.93 [0.87, 0.98] | 1.00 |
| GB_noisy | 0.93 [0.88, 0.98] | 1.00 |

Every method finds the real duplicates. The errors are links between
different people, and the models trained on the noisy labels make the most of
them.

## Secondary: rule-cleaned labels, all 30,700 pairs

AUC change: name_age +0.000, B2 −0.017, LR −0.036, GB −0.039 (all intervals
tight; B2 and learners exclude zero). Ranking unchanged. F1 at the ICE-ID
threshold falls from 0.85-0.92 to 0.68-0.71.

## Reading

1. The noise does not reorder these four methods robustly. It erases the
   learned models' advantage over a hand rule: that advantage exists only
   against the noisy labels.
2. Reported F1 around 0.9 on this task is mostly agreement with over-merges.
   The drop to about 0.3 against audited labels follows mechanically from
   the noise rate, so it is a statement about the benchmark, not a new fact
   about the methods.
3. Tuning or training on these labels teaches a method to link different
   people: the learners link 93% of the audited two-human pairs.

## Limits

19 true positives in the audited condition, so F1 and AP there are very
uncertain. Negatives were not audited (some may be unlinked duplicates).
Methods are simple; Ditto, Splink and the companion paper's models were not
run. One auditor; internal evidence only.
