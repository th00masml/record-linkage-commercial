# Data licenses

## 1. Source article (public domain)

Francis Galton, *Finger Prints* (London: Macmillan, 1892). Public domain
(pub. 1892; author d. 1911). https://archive.org/details/fingerprints00galt
Used only for the quoted problem statement.

## 2. Dataset (CC-BY 4.0)

ICE-ID: A Historical Census Dataset for Longitudinal Identity Resolution
(Carvalho et al., 2025). **CC-BY 4.0** (stated in the dataset's
croissant.json / paper; arXiv:2506.13792).
- HuggingFace: https://huggingface.co/datasets/goldpotatoes/ice-id
- GitHub: https://github.com/IIIM-IS/ICE-ID-2.0
- Files used: raw_data/people.csv, parishes.csv, counties.csv.
- Underlying census records are Icelandic historical records (1703-1920),
  themselves public domain; the value added by the dataset (normalization,
  expert person identifiers) is CC-BY.
- Attribution: Carvalho, G.H. de, et al., ICE-ID (CC-BY 4.0). Derived
  files here (pairs, model scores) are shared under CC-BY 4.0.

## 3. Models

- `qwen2.5:14b` via Ollama (local).
- `claude-sonnet-5` via the authenticated Claude CLI (`claude -p`), OAuth
  session, **no API key**. Single-sample (CLI exposes no temperature);
  raw outputs cached under `outputs/` so metrics recompute without
  re-querying.

## Not committed

`data/raw/` (the ~122 MB people.csv and lookups) is gitignored; it is
CC-BY and re-downloadable from the URLs above.
