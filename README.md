# record-linkage-commercial

Can an LLM decide whether two historical census records are the same
person — better than a hand-built rule? Tested on Icelandic census data,
where patronymic naming means many different people share the same name
(Galton, 1892: identity is hardest "where there is but little variety of
surnames").

## Outcome: KILLED-AT-4

A competent deterministic linker (birth-year, parish, farm, partner,
geographic distance) scores AUC 0.905 on the preregistered 200 pairs. The
commercial frontier model (claude-sonnet-5) reaches only 0.816, the local
qwen2.5:14b 0.787. The rule is significantly better, and the frontier model is
not significantly better than the local one.

Post-hoc, on 4,000-pair held-out sets with the v1 leaks removed, the same
frozen rule (AUC 0.961) matches a Fellegi-Sunter model fitted with Splink
(0.959) and a dev-fitted logistic regression (0.963). claude-sonnet-5, called
without Claude Code's own context, ties the rule on 400 leak-free pairs (0.972
vs 0.969) and still trails it on the preregistered pairs (0.856 vs 0.905).
No LLM comes near the +0.05 margin. On clean structured fields, use a rule.

- Manuscript draft: `MANUSCRIPT.md` (bibliography: `references.bib`)
- Problem, source (Galton 1892, PD), preregistration + post-hoc notes: `PROBLEM.md`
- Numbers and verdict: `RESULTS.md`
- Licenses, data hash, model/tool versions: `DATA-LICENSES.md`; code is MIT (`LICENSE`)

## Run it

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-extra.txt
# data: ICE-ID (CC-BY 4.0) from https://huggingface.co/datasets/goldpotatoes/ice-id
#   raw_data/people.csv, parishes.csv, counties.csv into data/raw/
python -m pytest tests

cd src
# v1 (preregistered; model scores are cached in outputs/)
python prepare_pairs.py
python baseline.py
python score.py local && python score.py comm     # needs Ollama qwen2.5:14b / authenticated `claude`
python report.py
python analysis_v1.py                             # leak diagnostics, LLM-as-feature stacking

# v2 (post-hoc; larger, leak-controlled, dev/test split by name)
python prepare_pairs_v2.py
python baselines_v2.py                            # B1, B2 frozen, Fellegi-Sunter (EM / counts), LR
python splink_baseline.py                         # Splink 5 Fellegi-Sunter
python report_v2.py
python stacking_v2.py                             # LLM as a feature on v2
```

claude-sonnet-5 runs on v2 and the bare-instrument v1 re-run are cached in
`outputs/`; qwen2.5:14b on v2 and few-shot runs still need a machine with
Ollama (commands in `RESULTS.md`).

## Use of AI tools

The research questions, the choice of what to test, the decisions at each
step and the architecture of the code are the author's; the author also wrote
and refactored code. Claude (Anthropic) was used as a research and coding
assistant under the author's direction: it wrote and ran analysis code,
proposed parts of the audit design, and drafted the documentation and the
note. The author is responsible for the content. Each pre-registration
protocol was committed before the analysis it governs.
