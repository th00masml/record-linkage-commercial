# Audit results: within-wave positives with conflicting parents

Protocol: `PROTOCOL.md` (committed before the sample was drawn).
Code: `evidence.py`, `run_audit.py`. Per-item evidence and verdicts:
`audit_sample.jsonl`. Internal evidence only (people.csv), no source images.

## Primary (pre-registered rule)

over_merge by rule 1: **35 / 50 = 0.70**, Wilson 95% CI [0.562, 0.809].
Lower bound >= 0.50, so by the protocol the critique stands: in this stratum
the `person` label joins two humans, it is not a kin-field slip.

Rule counts: over_merge 35, undetermined 11, duplicate_row 2,
kin_link_error 2.

## Manual reading of the 15 non-rule items (reported separately)

| final | n |
|---|---|
| over_merge | 12 |
| probable_over_merge | 1 (#20, same farm entry, two mothers) |
| duplicate_row | 1 (#43, one human, kin error) |
| undetermined | 1 (#25, possibly a grandparent double listing) |

Rule + manual: 47 / 50 over_merge, Wilson 95% CI [0.838, 0.979]
(48 / 50 with the probable one). This number includes judgment calls by the
auditor and is not the pre-registered result.

## Why the rule under-counts

The `father` field often points to a stepfather, foster father or household
head ("hennar barn", "fósturson", "uppeldisdóttir"), so the patronym test
fails and the row is not "anchored". Both kin links and person links are
noisy; the rule only credits cases where kin links are clean.

## Patterns

- Different patronyms under one person ID in the same census (#11 Pálsdóttir
  vs Narfadóttir, #41 Guðlaugsson vs Steingrímsson, #7 Bergsteinsson vs
  Þórðarson). Dataset-wide, 331 persons have two patronym stems inside one
  wave (heuristic, 4-letter stem; some spelling variants slip through).
- Over-merges persist: the same ID carries two parallel rows across several
  waves (#45 Guðrún Jónsdóttir b.1864: two rows in 1880, 1901, 1910, both
  "kona hans" in 1901 and 1910). One ID, two life courses.
- Distances: same parish up to 465 km apart (#17) in one census.

## Scope

This stratum only: 859 of 10,288 within-wave positive pairs (578 persons).
Persons with a parent conflict or a patronym conflict inside one wave: 834
(union; overlap 75). The other within-wave positives are not audited here.
