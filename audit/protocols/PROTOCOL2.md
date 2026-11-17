# Audit 2 protocol: random within-wave positives outside the parent-conflict stratum

Fixed before the sample was inspected. Calibration used 6 pairs drawn with
seed 12345 that are not in the sample. The sample indices (seed 0) were drawn
once only to exclude them from calibration; their contents were not printed.

## Population and sample

ICE-ID within-wave positive pairs (two rows, one wave, same `person`): 10,288.
Audit 1 covered the 859 with conflicting father or mother IDs. Population
here: the other 9,429 pairs. Unit = pair (the benchmark's evaluation unit).
Sample: 100 pairs, `random.Random(0).sample(remaining, 100)`.

## Evidence (people.csv only)

Patronym stems (father-name stems sharing fewer than 3 leading letters count as
different), marital status codes (Ó never married, G married, E widowed,
S separated), partner IDs, spouse-type status (kona hans, bóndi, húsbóndi,
húsfreyja, ...), distance between parishes on census day, same farm, birth-year
difference, away/visitor markers in status (staddur, heimili, gestur,
aðkomandi, fjarverandi, ferð, vertíð, sjóróðr, nám-/skóla-, lækn-), and the
other waves in which the same ID also has two rows (parallel lineage).

## Verdict rules (in order)

1. **over_merge** (two humans) if any of:
   a. patronym stems differ;
   b. marital conflict: one row Ó and the other G, E or S;
   c. both rows spouse-type status, and partner IDs differ or parishes > 20 km apart;
   d. the ID has two rows in at least 2 other waves, parishes > 20 km apart,
      and neither row carries an away/visitor marker.
2. **duplicate_row** (one human, label correct): same farm, birth years within 1.
3. **double_listing** (one human listed at two places, label plausibly correct):
   an away/visitor marker on either row, or parishes within 5 km.
4. **undetermined**: everything else. Manual reading, reported separately,
   never pooled into the rule count.

## Outcomes

Primary: share over_merge by rule, Wilson 95% CI.
Secondary: stratified estimate over all 10,288 within-wave positives,
combining audit 1 (rule: 35/50) and audit 2 (rule), weights 859 and 9,429,
normal-approximation 95% CI.

Decision: if the stratified lower bound is at or above 0.10, the claim
"at least one in ten within-wave positives joins two humans" goes into the
note. Below 0.10, the note restricts the claim to the conflict stratum.
