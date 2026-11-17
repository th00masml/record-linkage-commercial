# Within-wave identity labels in ICE-ID join different people: an audit and its consequences for evaluation

*Draft, 2026-11-17. All analyses are in `audit/` of this repository; each
audit and analysis protocol was committed before its sample was inspected or
its methods were scored (commits 73fdd21, d72ee7f, c938383, 73bbddf).*

## Abstract

ICE-ID (Carvalho et al., arXiv:2506.13792) releases 984,028 Icelandic census
records (1703-1920) with 226,864 expert-curated person identifiers and defines
two linkage tasks, within-wave and across-wave. We audit the positives of the
within-wave task: 10,288 pairs of records that share a person ID inside one
census. Using only evidence internal to the released table (patronyms, marital
status, household position, kin links, distance on census day, and repeated
duplication across waves), pre-registered rules classify 46% of these pairs as
two different people (stratified estimate, 95% CI 37-55%); a manual reading of
the same 150 audited pairs raises this to 72% (64-80%). Scoring four methods
and Splink against ICE-ID labels and against audited labels, we find that the
noise does not robustly reorder methods, but it creates an advantage of
supervised models over a hand-written rule (+0.03 AUC) that disappears under
audited labels, and that any threshold tuned on ICE-ID labels links 87-99% of
the pairs the audit identifies as two people. Reported precision and F1 on
the within-wave task should not be read as accuracy until the labels are
repaired.

## 1. What the dataset paper says about labels

The labelling procedure is described in one sentence: "Expert-curated using
genealogical records and parish registers" (Section 13.3). No annotator
count, procedure, agreement measure or audit is reported. The limitations
section lists "Label noise: Unknown amount of transcription error rates"
(Section 14), which concerns transcription, not linkage. Over-merges (one ID
for two people) and under-merges (two IDs for one person) are not discussed.

The within-wave task is defined in Section 12.1: "determine whether two
records from the same census wave refer to the same individual." Positives
are all pairs sharing a person ID; negatives are sampled two per positive
from records with different IDs in the same blocking partition (Section 12.3).

## 2. The population

10,288 within-wave positive pairs, from 6,356 persons. Simple counts already
look wrong for a census taken on a single day:

- 4,143 pairs (40%) sit in parishes more than 50 km apart; only 559 (5%)
  share a farm.
- 859 pairs carry different father or different mother IDs.
- 331 persons have two different patronyms inside one wave, i.e. two
  different fathers by name.
- Only 83 pairs have a status that marks a visitor or an absent person, the
  legitimate reason for one person to appear twice.

## 3. Audit

Evidence is internal to `people.csv`: no source images were consulted.

**Audit 1, conflict stratum** (859 pairs with conflicting parent IDs; 50
persons drawn at random). Rule: both records live with their own linked
parents on different farms, with plausible parent ages and consistent
patronyms. Result: 35/50 over-merges (Wilson 95% CI 0.56-0.81). Manual
reading of the remaining 15 adds 12 (47/50, 0.84-0.98). The rule
under-counts because the `father` field often points to a stepfather or
foster father.

**Audit 2, everything else** (9,429 pairs; 100 drawn at random). Rules:
different patronyms; never-married versus married or widowed on the same day;
two spouse-type statuses in two households; or the ID duplicated in at least
two other waves, more than 20 km apart, with no visitor marker. Result:
44/100 over-merges (0.35-0.54). Manual reading of the 52 pairs the rules did
not classify: 70 over-merges, 18 correct labels, 12 undetermined. The rules
are conservative: their "double listing" class let through 10 pairs that are
two people, because parish centroids put different farms of one parish at
0 km.

**Combined.** Stratified over all 10,288 pairs: 46% over-merges by the
pre-registered rules (37-55%), 72% by manual reading (64-80%).

**Examples** (person ID, wave):

- 6951, 1840: Jón Þorgrímsson, foster son at Eydalir í Breiðdal, and Jón
  Þorgrímsson, son of Þorgrímur Jónsson at Stóri-Laugardalur í Tálknafirði,
  465 km apart.
- 54886, 1840: Hólmfríður Pálsdóttir, foster daughter at Dagverðarnes, and
  Hólmfríður Narfadóttir, daughter of Narfi Þorleifsson at Helgafell. Two
  patronyms, two fathers; the pair is duplicated again in 1835, 1845 and 1850.
- 52871, 1850: Sigríður Bjarnadóttir, unmarried servant at Hrafnagil í
  Eyjafirði, and Sigríður Bjarnadóttir, married wife at Dyrhólar í Mýrdal,
  245 km apart; the ID carries two records in six other waves.
- 143596: Guðrún Jónsdóttir b. 1864 has two records in 1880, 1901 and 1910;
  in 1901 and 1910 both are married women in separate households.

**What the correct positives look like.** The 18 correct labels are mostly
de jure / de facto double enumeration: a student in Reykjavík also listed at
home, a seasonal fisher listed at the fishing station and at home, a visitor
("aðkomandi"). These are what the within-wave task should contain. They are a
minority of its positives.

## 4. Consequences for evaluation

We built the within-wave evaluation set as Section 12.3 prescribes (10,288
positives, 20,412 same-block negatives) and scored: a name-and-age matcher;
the hand rule B2 (birth year, parish, county, farm, distance); logistic
regression and gradient boosting trained on ICE-ID labels (5-fold, grouped
by person), with and without curated kin-ID features; and Splink 4
(Fellegi-Sunter, EM, no labels, no curated IDs). Each method was scored on
the same pairs under ICE-ID labels and under audited labels (137 audited
positives, 118 of them two people, plus 300 negatives, weighted by stratum).

| method | AUC, ICE-ID labels | AUC, audited labels | change [95% CI] |
|---|---|---|---|
| name and age | 0.835 | 0.872 | +0.037 [-0.075, +0.130] |
| hand rule B2 | 0.948 | 0.949 | +0.001 [-0.040, +0.039] |
| logistic regression | 0.976 | 0.942 | -0.034 [-0.079, +0.005] |
| gradient boosting | 0.980 | 0.945 | -0.036 [-0.082, +0.002] |
| Splink (no labels) | 0.917 | 0.929 | +0.011 [-0.035, +0.054] |

Three findings.

1. **The supervised advantage is an artefact of the labels.** Gradient
   boosting beats B2 by +0.032 AUC [+0.011, +0.057] under ICE-ID labels and by
   -0.004 [-0.029, +0.017] under audited labels; logistic regression +0.028
   versus -0.007. Removing the curated partner, father and mother ID features
   leaves this unchanged (+0.031 versus -0.007). The method ranking itself
   does not change beyond the uncertainty.
2. **Tuning on the labels teaches methods to link different people.** At the
   threshold that maximises F1 under ICE-ID labels, the share of audited
   two-person pairs each method links is 0.67 (name and age), 0.87 (B2), 0.93
   (both learners) and 0.99 (Splink). All methods except the name matcher find
   every true duplicate. Splink at its own threshold (match probability 0.5)
   links 5% of two-person pairs but only 23% of true duplicates.
3. **Reported F1 is mostly agreement with over-merges.** F1 at the ICE-ID
   threshold is 0.80-0.92 against ICE-ID labels and 0.24-0.35 against audited
   labels. This drop follows from the noise rate and is very uncertain (19
   true positives in the audited sample); it says more about the benchmark
   than about the methods.

A secondary analysis on all 30,700 pairs with labels cleaned by the audit
rules shows the same direction (learners -0.035 to -0.040 AUC, B2 -0.017,
Splink -0.007) but is partly circular, since the cleaning rules use record
features the learners can see.

## 5. Recommendations

1. Flag or split persons whose within-wave duplicates conflict on patronym,
   marital status, parents or household; the rules in `audit/` are a starting
   point, not a substitute for the curators' sources.
2. Release a per-link confidence or provenance field, and document how
   identifiers were assigned (who, which sources, any agreement checks).
3. Until then, evaluate the within-wave task only on an audited subset and
   report results on it separately from the across-wave task.
4. State how unlabelled records (about half the table) enter negative
   sampling.
5. Treat `partner`, `father` and `mother` as curated labels, not record
   features: they hold person IDs.

## 6. Limitations

One auditor, no second rater. Internal evidence only; source images and
parish registers were not consulted, and the curators may hold evidence we do
not. The across-wave task was not audited, nor were negatives (some may be
unlinked duplicates). Methods are simple; Ditto, the companion paper's models
and its reported results were not examined, because the companion paper was
not available to us. The evaluation unit is the pair, so persons with many
duplicate records weigh more, as they do in the benchmark.

## Reproduction

`audit/PROTOCOL*.md` (pre-registrations), `audit/RESULTS*.md`,
`audit/run_audit.py`, `audit/run_audit2.py`, `audit/noise_effect.py`,
`audit/splink_within_wave.py`. Data: ICE-ID `raw_data/people.csv`,
`parishes.csv`, `counties.csv` from Hugging Face (goldpotatoes/ice-id).
