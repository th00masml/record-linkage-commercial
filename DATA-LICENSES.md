# Licenses and provenance

## Code

MIT, see `LICENSE`. Covers `src/` and `tests/`.

## 1. Source article (public domain)

Francis Galton, *Finger Prints* (London: Macmillan, 1892). Public domain
(pub. 1892; author d. 1911). https://archive.org/details/fingerprints00galt
Used only for the quoted problem statement.

## 2. Dataset (CC-BY 4.0)

ICE-ID: A Novel Historical Census Dataset for Longitudinal Identity Resolution
(Carvalho et al., arXiv:2506.13792; v1 2025-06-11, v2 2026-02-23).
- HuggingFace: https://huggingface.co/datasets/goldpotatoes/ice-id
- GitHub: https://github.com/IIIM-IS/ICE-ID-2.0
- Files used: `raw_data/people.csv`, `parishes.csv`, `counties.csv`.
  `people.csv` SHA-256 (= its LFS oid):
  `81e30e5ce6fc649cb077bb0899caaa7cb0ca8ee8743bb5f44855c606395d5c7f`
  (121,919,888 bytes, downloaded 2026-11-17; the v1 run used the same file,
  since `prepare_pairs.py` regenerates `data/derived/pairs.jsonl` byte for byte).
- License: **CC-BY 4.0**, as stated in the dataset's `croissant.json`
  (`"license": "https://creativecommons.org/licenses/by/4.0/"`) and in the
  paper (v2 §13.4). Note: the HuggingFace README's front matter says
  `license: mit`. The two upstream statements conflict; we follow the more
  restrictive and more specific one (CC-BY 4.0, which requires attribution)
  and attribute accordingly.
- Underlying census records are Icelandic historical records (1703-1920),
  themselves public domain; the value added by the dataset (normalization,
  expert person identifiers) is CC-BY.
- Attribution: Carvalho, G. H. de, et al., ICE-ID (CC-BY 4.0). Derived files
  here (`data/derived/`, model scores in `outputs/`) are shared under CC-BY 4.0.

## 3. Models and tools

What is and is not on record for the v1 run (2026-08-09):

| instrument | identifier used | recorded? |
|---|---|---|
| local | `qwen2.5:14b` via Ollama, temperature 0, seed 7, `num_predict` 8 | Ollama version and model digest were **not** logged |
| commercial | `claude-sonnet-5` alias via `claude -p`, OAuth session, no API key | CLI version and the dated model snapshot behind the alias were **not** logged |

These gaps cannot be filled after the fact, and we do not guess them. The CLI
exposes no temperature, and `--append-system-prompt` added our instruction to
Claude Code's own default system prompt, so the commercial instrument was
"Claude Code in print mode", not a bare model call.

From now on every non-v1 `score.py` run writes `outputs/<run>/run_meta.json`
with: the Ollama server version and model digest (`/api/version`, `/api/tags`),
or the `claude --version` string and, with `--json-output`, the model ids the
CLI reports per call, or, with `--backend api`, the exact model id the API
returns in each response.

Python packages used for the v2 analysis (2026-11-17): Python 3.13.18,
numpy 2.5.2, splink 5.0.0, duckdb 2.0.0, pytest 9.2.0
(`requirements.txt`, `requirements-extra.txt`).

## Not committed

`data/raw/` (the ~122 MB people.csv and lookups) is gitignored; it is CC-BY
and re-downloadable from the URLs above.
