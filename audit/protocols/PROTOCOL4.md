# Protocol 4: Splink on the within-wave task, and a kin-ID leakage check

Fixed before any Splink score was computed. Same evaluation pairs, label
conditions, weights and metrics as `PROTOCOL3.md`.

## Splink (Fellegi-Sunter, no labels)

Splink 4 on DuckDB, `dedupe_only`, trained on all labelled rows.
Comparisons: first name (exact, Jaro-Winkler ≥ 0.9), patronym (exact,
Jaro-Winkler ≥ 0.9), birth year (exact, ±1, ±2, else), parish, county, farm
(exact), distance between parishes (1, 10, 50 km), partner name (exact; name
read from the `source_partner` row of the transcription, never from a curated
person ID), marital status (same, never-married vs ever-married, else).
No person, partner, father or mother ID is used anywhere.

Training: u by random sampling (1e7 pairs, seed 0); m by EM in two sessions,
blocked on wave + first name + patronym, and on wave + birth year + parish.
Prior from the deterministic rule "same wave, full name, birth year and
parish" with recall 0.8 (affects only the 0.5 threshold, not AUC).
Scoring: the 30,700 evaluation pairs exactly, via a pair-id link.

Score = match weight. Two thresholds: F1-max under ICE-ID labels (as for the
other methods) and Splink's own label-free threshold, match probability ≥ 0.5.

## Leakage check

The PR #1 review found that ICE-ID's `partner` column carries curated person
IDs. The protocol 3 methods used partner, father and mother person-ID
equality. Rerun: B2 without the partner term, LR and GB without the three
kin-ID features, everything else unchanged.

## Expectations, stated before scoring

E3: Splink loses less AUC under audited labels than LR_noisy and GB_noisy.
E4: at match probability ≥ 0.5, Splink links a smaller share of the audited
two-human pairs than LR_noisy and GB_noisy at their ICE-ID thresholds (0.93).
E5: without kin-ID features, the learners still lead B2 under ICE-ID labels
and lose that lead under audited labels.
