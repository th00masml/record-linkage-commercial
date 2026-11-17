# Protocol 3: does the label noise change method results on the within-wave task?

Fixed before any method was scored.

## Evaluation pairs (ICE-ID protocol, section 12.3)

Positives: all 10,288 within-wave pairs sharing a person ID. Negatives: two per
positive, same wave, same block (same first name + patronym; fallback same
first name), different person IDs, both rows labelled. Seed 0.

## Methods (none built from the audit)

- **name_age**: 2·[same full name] + [same birth year] + 0.5·[birth years 1 apart].
- **B2**: the hand rule from `src/baseline.py` (birth year, parish, county,
  farm, partner, distance), written before the audits.
- **LR_noisy**, **GB_noisy**: logistic regression and histogram gradient
  boosting on generic record features (birth-year gap, same name, parish,
  county, farm, log distance, partner/father/mother same or different,
  marital codes same or conflicting, patronym stems different, status type
  match, visitor marker). Trained on ICE-ID labels, 5-fold out-of-fold
  scores grouped by person. No feature uses person IDs.

## Label conditions, same pairs

1. **ICE-ID**: every within-wave pair is a positive.
2. **Audited (primary)**: the 150 manually read pairs; over_merge and
   probable_over_merge become negatives, label_correct stays positive,
   undetermined pairs are dropped from both conditions. Weights: audit 1
   pairs 859/50, audit 2 pairs 9,429/100, plus 300 random negatives with
   weight 2·10,288/300.
3. **Rule-cleaned (secondary, partly circular)**: all pairs; positives that
   the pre-registered audit rules call over_merge become negatives. Methods
   that read marital or patronym features share signals with these rules, so
   this condition is reported only as a check on direction.

## Metrics

Weighted ROC-AUC and average precision; F1 at the threshold that maximises F1
under ICE-ID labels, then held fixed under the other labels. 95% CIs from
1,000 stratified bootstrap resamples, for each AUC and for the change in AUC.

## Expectations, stated before scoring

E1: name_age and the two noisy-trained learners lose AUC under audited
labels; B2 gains, because it penalises pairs far apart on census day.
E2: the method ranking under ICE-ID labels differs from the ranking under
audited labels. Results are reported whether or not E1 and E2 hold.
