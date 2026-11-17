---
title: "Wiki Log"
type: synthesis
created: 2026-04-15
updated: 2026-04-15
status: review
author: llm
---

# Wiki-Log

## [2026-04-15] meta | Projekt initialisiert
- Ordnerstruktur angelegt
- Schema-Dokument (CLAUDE.md) erstellt
- Index und Log initialisiert

## [2026-11-17] plan | gray-zone-triage pre-registered
- Design approved: input/ideas/gray-zone-triage-design.md
- Plan pre-registered: input/ideas/gray-zone-triage-plan.md (H1: R >= 0.20, alpha = beta = 0.05)
- Known gap at pre-registration: ICE-ID companion methods paper not retrieved (search credits exhausted, arXiv blocked by proxy)

## [2026-11-17] data | gray-zone-triage Task 1 done
- ICE-ID raw files fetched and fingerprinted (input/data/README.md); row count and label rate match the dataset paper
- Session network switched to the saved `record-linkage-commercial` environment to reach huggingface.co

## [2026-11-17] review | gray-zone-triage Task 2 reviews
- Spec review: PASS (independent rebuild reproduces all 1,194,338 pairs and the split). Format differs from plan: `candidates.csv` + `records.jsonl` joined on rid instead of one `candidates.jsonl` with inline records (storage only; inline records would be several GB). Script appends large derived files to `.gitignore`.
- Edge case: one record has raw sex `KArl`; the case-sensitive exp-1 SEX map leaves it unblocked (1 anchor, 1 true pair lost).
- Quality review: SOUND WITH ISSUES. Verified in main context:
  - 2026-11-17 · deviation-candidate · plan=gray-zone-triage · `partner` is the spouse's expert-curated `person` id (99.67% equal to person of the `source_partner` row; filled for 13.2% of labelled vs 0.1% of unlabelled records). B2's cross-wave partner rule therefore reads the gold-standard linkage of the spouse. Fires on 2.1% of candidate pairs, 10% of positives. `farm` is a place id (filled for 99.8% of unlabelled records), not leakage.
  - Experiment 1 impact: 4 of 200 pairs had a partner match; B2 AUC without partner 0.900 [0.856, 0.937] vs 0.905; KILLED-AT-4 unchanged.
- No thresholds computed, no LLM run, no test outcome inspected yet. Awaiting researcher decision before Task 3.
