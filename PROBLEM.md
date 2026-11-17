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
  geography, household, and life-course do. [Not quite true: see note 3.]

Both models and baselines see the same fields (parish/county given by
name and coordinates).

## Primary metric, baselines, kill criterion (fixed before any model run)

Primary metric: **ROC-AUC** (same person = 1), chance 0.50, 95% CI by
bootstrap (1000 resamples, seed 0).

Baselines (run first):
- B1 = birth-year agreement only.
- B2 = a competent hand-built rule (birth-year, parish, county, farm,
  partner, geographic proximity via coordinates). Measured: **B1 0.72,
  B2 0.905** on this set. [Provenance of the B2 weights: see note 1.]

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

[Superseded by note 4: the claim below is too strong.]

Not a threat here, and this is the analysis, not an omission. The task is
to link anonymized census rows whose true identity groupings are a private
expert curation over internal ID codes; there is no public answer key
tying these individual records together that a model could have
memorized. The names are common Icelandic patronymics shared by thousands.
Any score therefore reflects reasoning over the shown evidence, not
recall. (A frontier model cannot know that census-row X in 1835 and
census-row Y in 1840 are the same farm laborer.)

---

## Post-hoc notes (added 2026-11-17, after review)

The text above is the preregistration as committed; it is left unchanged
except for the bracketed pointers. Nothing below moves a threshold or changes
the v1 verdict.

1. **B2 weights.** The weights (2/1 birth year, 1.5 parish, 0.5 county,
   1 farm, 2 partner, distance decay to 100 km) are hand-set. The repository
   has a single snapshot commit, so there is no timestamped evidence that they
   were fixed before the 200 test pairs were seen, and we do not claim it. The
   check that can still be made was made: the same frozen weights were applied
   to 4,000 freshly and randomly drawn test pairs, and compared with models
   whose weights were fitted on a separate dev split with no names in common
   with the test split (Fellegi-Sunter by EM and by counts, Splink, logistic
   regression). Frozen B2 scores AUC 0.961 there,
   the dev-fitted logistic regression 0.963, Splink 0.959 (RESULTS.md, v2). The
   rule is not overfitted to the v1 pairs.
2. **`partner_id` is gold-derived.** In ICE-ID `people.csv`, `partner` holds the
   partner's expert-curated *person* id (65,186 of 65,404 values equal the
   `person` of a co-resident row). Equal partner ids across waves therefore
   encode the expert linkage of the partner. It affects 4 of 100 positives;
   dropping the term moves B2 from 0.905 to 0.900. v2 uses the partner's name.
3. **Names and census years carry class signal in v1.** 20 of 100 negatives
   differ in full name (middle names), against 1 of 100 positives; negatives
   come from any two waves while positives are adjacent appearances, so the
   census gap alone has AUC 0.831 on v1; and 16 negatives have record A in the
   later wave, positives never. Selection was "first N in sort order", so all
   100 negatives have first names beginning with "A". v2 fixes all four.
4. **Memorization.** ICE-ID, including the `person` column, has been public on
   HuggingFace since mid-2025, so a model trained on web data after that date
   could in principle have seen the answer key. Recall of specific person ids
   from 984k rows is implausible, and the models are not shown the ids, but
   contamination is not excluded "by construction". See RESULTS.md.
