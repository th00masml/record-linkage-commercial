# record-linkage-commercial

Can an LLM decide whether two historical census records are the same
person — better than a hand-built rule? Tested on Icelandic census data,
where patronymic naming means many different people share the same name
(Galton, 1892: identity is hardest "where there is but little variety of
surnames").

## Outcome: KILLED-AT-4

A competent deterministic linker (birth-year, parish, farm, partner,
geographic distance) scores AUC 0.905. The commercial frontier model
(claude-sonnet-5) reaches only 0.816, the local qwen2.5:14b 0.787 — the
rule is significantly better, and the frontier model is not significantly
better than the local one. On clean structured fields, use a rule.

- Problem, source (Galton 1892, PD), preregistration: `PROBLEM.md`
- Licenses (source PD; ICE-ID CC-BY 4.0): `DATA-LICENSES.md`
- Numbers and verdict: `RESULTS.md`

## Run it

```
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
# data: ICE-ID (CC-BY) from https://huggingface.co/datasets/goldpotatoes/ice-id
#   raw_data/people.csv, parishes.csv, counties.csv into data/raw/
ollama pull qwen2.5:14b
# commercial instrument uses the authenticated `claude` CLI

cd src
python prepare_pairs.py
python baseline.py
python score.py local && python score.py comm
python report.py
```
