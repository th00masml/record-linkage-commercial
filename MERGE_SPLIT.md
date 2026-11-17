# Merge / split items for the look-or-ask agent study

Source items for carrying the act/look/ask grid of Zhao et al. (arXiv
2611.17639) over to referent tracking. Built by `src/merge_split_pairs.py`
(deterministic, seed 0, no model calls) from ICE-ID raw `people.csv`.
Output: `data/derived/merge_split.jsonl`, counts in
`data/derived/merge_split_summary.json`.

## Design

Every item is two census rows that answer the same name lookup: first name +
patronym + sex, birth year within ±2 (covers 99.6% of adjacent-wave drift
for linked persons). The lookup window holds exactly those two rows, counting
unlabelled rows too, so K is complete within the fixture database. Database =
waves 1835–1860 (contiguous, 45–92% of rows carry an expert person ID).

| class | rows | persons | K | n | label strength |
|---|---|---|---|---|---|
| merge | 2, different waves | 1 | 1 | 174 | curator_link (positive expert judgment) |
| split | 2, different waves | 2 | 2 | 174 | 19 kin_differs, 155 unlinked_only |
| distinct | 2, same wave | 2 | 2 | 115 | same_wave (two places in one census) |

Merge and split are matched cell by cell on wave gap, birth-year difference,
same parish, same farm. The surface of the two records cannot separate them;
only the hidden person ID does. `distinct` is the paper's "second same-name
customer" control. `name_universe_n` is the size of the name-only lookup
(birth year ignored), for building count/page conditions.

## Cleaning, and why split is the weak class

- 6,356 persons appear twice in one wave (e.g. person 100123 in two parishes
  with two fathers, 1840–1860): curator over-merges. Dropped, with every
  window that touches them.
- `partner`, `father`, `mother` hold person IDs, not row IDs.
- Split/distinct pairs that share a farm or a parent or partner ID under two
  person IDs are almost surely one human the curators did not link: 222 of
  545 split candidates (41%). Dropped.
- Even after that, "different person ID" means "not linked", not "two
  people". Example: Kristín Þorvarðardóttir b. 1819, child at
  Breiðabólsstaður 1835, wife at Teigur 6.6 km away 1855, two IDs. Only
  `kin_differs` (different parent IDs) and `distinct` are trustworthy K = 2.
  Treat `unlinked_only` as a sensitivity set, not ground truth.

## Side finding for KILLED-AT-4

Of the 100 hard negatives in `data/derived/pairs.jsonl`, 11 share a farm and
7 share a parent or partner ID. Some negatives are probably the same person,
which penalizes every method. The verdict is unlikely to flip (the rule led
by 0.089 AUC), but the negative set should be re-filtered before reuse.

## Not done yet

Fixtures: the four lookup modes (one / count / page / all), placement of the
target row in the result list, the user claim per item, and the tool that
reveals identity on look.
