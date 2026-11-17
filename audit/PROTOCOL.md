# Audit protocol: within-wave positives with conflicting parents

Fixed before the audit sample was drawn. Calibration used 4 pairs drawn with
seed 12345 (not reused); everything below was written after seeing only those.

## Population and sample

ICE-ID within-wave positive pairs (two rows, one wave, same `person`) where
both rows carry a father ID and the IDs differ, or both carry a mother ID and
they differ: 859 pairs, 578 persons. Sample unit = person (one pair per
person, the first in row order), 50 persons, `random.Random(0)`.

## Evidence (internal to people.csv; no source images)

Per row: linked father and mother (person IDs), whether each is enumerated in
the same wave on the same farm (co-resident), parent-child age gap, whether
the patronym matches the linked father's first name (prefix heuristic), the
farm household. Per pair: distance between parishes, same farm, and the
person's appearances in other waves.

A row is **anchored** when at least one linked parent is co-resident, every
linked parent has a plausible age gap (father 15-70, mother 14-55, unknown
counts as plausible), and the patronym check is not False.

## Verdict rules (applied in this order)

1. **over_merge**: both rows anchored, different farms, and the sets of
   co-resident parents are disjoint. Two children, each living with their own
   parents: the `person` label joins two humans.
2. **duplicate_row**: same farm and birth years within 1. One household
   listed twice; the parent conflict is a kin-field error.
3. **kin_link_error**: exactly one row fails anchoring because of an age gap
   or patronym failure, and the rows are otherwise compatible (parishes within
   5 km, birth years within 1).
4. **undetermined**: anything else. These get a manual reading, reported
   separately and marked `manual`, never pooled into the rule count.

## Outcome and decision

Primary: share of the 50 classed `over_merge` by rule 1, Wilson 95% CI.
If the lower bound is at or above 0.50, label errors (not kin-field errors)
are the main cause of this stratum, and the within-wave critique stands.
If it is below 0.50, the critique is about kin fields, not identities, and
it must be rewritten.

Scope: this stratum only (859 of 10,288 within-wave pairs). No extrapolation
to the other 9,429 pairs from this audit.
