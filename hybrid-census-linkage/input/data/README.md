# Input data

## ICE-ID raw release (Task 1, gray-zone-triage plan)

Source: https://huggingface.co/datasets/goldpotatoes/ice-id (`raw_data/`), revision `main`.
License: CC-BY 4.0. Cite de Carvalho et al. 2025, arXiv 2506.13792.
Downloaded: 2026-11-17 into `input/data/raw/` (gitignored; re-fetch with the URLs below).

| File | Rows | Bytes | SHA-256 |
|---|---|---|---|
| `people.csv` | 984,028 | 121,919,888 | `81e30e5ce6fc649cb077bb0899caaa7cb0ca8ee8743bb5f44855c606395d5c7f` |
| `parishes.csv` | 464 | 32,972 | `0116a146b32b67205567285f0a63b4e34cc3954b7a301620c59543bf0f8223a2` |
| `counties.csv` | 385 | 26,714 | `ef0b726f79a0e23148e699666b20a4f9178d9d34fbf9f76f95bb2d5205171c29` |

Checks against the dataset paper: row count 984,028 matches; records with a
non-empty `person` identifier = 493,681 (50.17%), matching the reported
average label rate of 50.17%. The `people.csv` SHA-256 equals the LFS oid
published on Hugging Face.

Re-fetch:

```bash
for f in people.csv parishes.csv counties.csv; do curl -sSL --fail -o input/data/raw/$f "https://huggingface.co/datasets/goldpotatoes/ice-id/resolve/main/raw_data/$f"; done
```
