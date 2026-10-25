# Are these two records the same person? (Icelandic census linkage)

## The original statement

Francis Galton, *Finger Prints* (London: Macmillan, 1892), Introduction,
pp. 14-16. Galton poses the identity-resolution question directly -- the
task of deciding, of two records,

> "whether or no the suspicion of their referring to the same person was
> justified"

and notes where it is hardest:

> "the need of some such system is shown to be greatly felt in many of
> our dependencies; ... where there is but little variety of surnames".

Source scan (public domain): Internet Archive,
https://archive.org/details/fingerprints00galt . Public domain: published
1892; Galton died 1911. See DATA-LICENSES.md.

Galton's answer was biometric (fingerprints). For historical population
records there are no fingerprints -- only names, ages, places, kin. And in
Iceland the naming is patronymic: "little variety of surnames" is the
normal condition, so the name alone almost never identifies a person.

## The bound and the capability

Labor. Linking a population across censuses means, for every record,
deciding which record in another year is the same human -- with names
that collide constantly, ages that drift, farms and parishes that change.
Genealogists do it by hand, one person at a time. The candidate capability
is **reasoning over messy, inconsistent structured evidence** (patronymic
logic, plausible migration given geography, life-course plausibility) at
scale -- and the question this run asks is whether an LLM adds anything a
plain deterministic rule does not.

## Still open

Historical record linkage "remains challenging due to name variation,
missing or inconsistent attributes, and changes in administrative
boundaries" -- ICE-ID (Carvalho et al., 2025, CC-BY), the benchmark used
here. Active, unsolved; large linkages are still expensive expert work.

## Data

ICE-ID (goldpotatoes/ice-id, CC-BY 4.0): 984,028 Icelandic census records,
1703-1920, with 226,864 expert-curated person identifiers. Fields: name
components, patronym, sex, birth year, status, farm/parish/county/district,
partner. See DATA-LICENSES.md.

## Pairs (deterministic)

- **Positive** (100): the same person (same expert identifier) in two
  different census waves, chosen as the two adjacent (smallest-gap) waves
  -- the realistic "link census t to t+1" task.
- **Hard negative** (100): two DIFFERENT people who share the same first
  name + patronym + sex and were born within 1 year, from different waves.
  Icelandic homonyms -- exactly Galton's "little variety of surnames".
  The name carries no signal for any pair; only birth-year consistency,
  geography, household, and life-course do.

Both models and baselines see the same fields (parish/county given by
name and coordinates).

## Primary metric, baselines, kill criterion (fixed before any model run)

Primary metric: **ROC-AUC** (same person = 1), chance 0.50, 95% CI by
bootstrap (1000 resamples, seed 0).

Baselines (run first):
- B1 = birth-year agreement only.
- B2 = a competent hand-built rule (birth-year, parish, county, farm,
  partner, geographic proximity via coordinates). Measured: **B1 0.72,
  B2 0.905** on this set.

Instruments (same fields, same prompt):
- M_local = qwen2.5:14b (local).
- M_comm = claude-sonnet-5 (commercial, via the authenticated CLI).

**Kill criterion (fixed now):** the commercial instrument must reach
AUC >= 0.65, with a 95% CI excluding 0.50, AND beat the best deterministic
baseline (B2) by >= +0.05 AUC. If it does not beat a hand-built rule by
that margin, the LLM adds nothing here: verdict KILLED-AT-4, stated
plainly. Secondary question (not part of the kill test): does the
commercial model beat the local 14B -- is the frontier call worth it?

No threshold moves after seeing the model numbers.

## Memorization

Not a threat here, and this is the analysis, not an omission. The task is
to link anonymized census rows whose true identity groupings are a private
expert curation over internal ID codes; there is no public answer key
tying these individual records together that a model could have
memorized. The names are common Icelandic patronymics shared by thousands.
Any score therefore reflects reasoning over the shown evidence, not
recall. (A frontier model cannot know that census-row X in 1835 and
census-row Y in 1840 are the same farm laborer.)
