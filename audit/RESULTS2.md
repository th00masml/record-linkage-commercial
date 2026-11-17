# Audit 2 results: random within-wave positives outside the conflict stratum

Protocol: `PROTOCOL2.md` (committed before the sample was inspected).
Code: `evidence2.py`, `run_audit2.py`. Items, verdicts, manual reasons:
`audit2_sample.jsonl`. Internal evidence only (people.csv).

## Primary (pre-registered rules)

over_merge by rule: **44 / 100**, Wilson 95% CI [0.347, 0.538].
Rules fired: marital conflict 18, two households 8, patronym 9, parallel
lineage 9. Other verdicts: double_listing 23, duplicate_row 4,
undetermined 29.

Stratified over all 10,288 within-wave positives (audit 1 rule 35/50,
weights 859 / 9,429): **0.462**, 95% CI [0.372, 0.552]. Lower bound is
above the pre-registered 0.10, so the claim "at least one in ten within-wave
positives joins two humans" goes into the note.

## Manual reading (reported separately)

All 29 undetermined and all 23 double_listing items were read; 9 rule
over_merges were spot-checked.

| final | n |
|---|---|
| over_merge | 70 (44 rule + 15 undetermined + 10 double_listing + 1 more from undetermined) |
| label_correct | 18 (4 duplicate_row + 14 read as one person listed twice) |
| undetermined | 12 |

Manual over_merge 70 / 100, Wilson [0.604, 0.781]. Stratified with audit 1
manual (47/50): 0.720 [0.638, 0.803]. Counting the doubtful rule verdict
(#35) as correct: 0.711 [0.628, 0.794].

## Where the rules went wrong

- Spot-check of rule over_merges: 8 of 9 hold. #35 (Þorsteinn Jónsson at
  Stöð, "son of the couple" and married householder on one day) may be a
  generational handover inside one household.
- double_listing was too generous: "parishes within 5 km" measures parish
  centroids, so two children on different farms of one parish pass. 10 of
  23 were two people (e.g. #64, Jón Jónsson b.1819, two workmen in Hruni,
  doubled in five consecutive waves).
- Marker list gaps: "sjóróandi" (#80, a seasonal fisher, one person) and
  Danish "huusbond" (#94) were not matched. Not fixed post hoc.

The rule estimate is therefore conservative.

## What a correct within-wave positive looks like

The 18 correct labels are mostly the de jure / de facto pattern: a person
enumerated where they were ("aðkomandi", "stud. art.", fishing season,
servant in town) and again at home. These are exactly what the within-wave
task should contain. They are a minority of its positives.

## Limits

One auditor, no second rater. Internal evidence only, no source images.
Unit is the pair, so persons with many duplicate rows weigh more, as they do
in the benchmark.
