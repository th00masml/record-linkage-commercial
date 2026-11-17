# ICE-ID within-wave label audit

Write-up: `../NOTE-ICE-ID.md`. Each protocol was committed before the analysis
it governs; the commits are listed at the top of the note.

## Layout

- `protocols/`: the four pre-registrations, unchanged since they were committed.
- `results/`: one results file per protocol.
- `scripts/`: the code.
- `data/`: script inputs and outputs (samples, verdicts, scores, numbers).

The files were moved into these folders after all four analyses were done.
In the pre-registration commits they sit directly in `audit/`. The move
changed only where the scripts look for `data/raw/` and for `audit/data/`.

## What belongs to which protocol

| # | question | protocol | results | scripts | data |
|---|---|---|---|---|---|
| 1 | Are within-wave positives with conflicting parents one person? | `PROTOCOL.md` | `RESULTS.md` | `evidence.py`, `run_audit.py` | in: `conflict_pairs.json`; out: `audit_sample.jsonl`, `rule_results.json` |
| 2 | Same question for random positives outside that stratum | `PROTOCOL2.md` | `RESULTS2.md` | `evidence2.py`, `run_audit2.py` | in: `remaining_pairs.json`; out: `audit2_sample.jsonl`, `rule_results2.json` |
| 3 | What the label noise does to method scores | `PROTOCOL3.md` | `RESULTS3.md` | `noise_effect.py`, `noise_effect_extra.py` (not pre-registered) | `noise_effect_results.json`, `noise_effect_scores.npz`, `noise_effect_extra.json` |
| 4 | Splink on the within-wave task, kin-ID leakage check | `PROTOCOL4.md` | `RESULTS4.md` | `splink_within_wave.py` | `noise_effect_results_p4.json`, `noise_effect_scores_p4.npz`, `splink_p4_extra.json`, `splink_within_wave_model.json` |

## Run it

From the repository root, with ICE-ID `people.csv` and `parishes.csv` in
`data/raw/`:

```
pip install -r requirements-audit.txt
python audit/scripts/run_audit.py
python audit/scripts/run_audit2.py
python audit/scripts/noise_effect.py
python audit/scripts/noise_effect_extra.py
python audit/scripts/splink_within_wave.py
```

Later scripts read the outputs of earlier ones, so keep this order.
`run_audit.py` and `run_audit2.py` rewrite `audit_sample.jsonl` and
`audit2_sample.jsonl` with the rule verdicts only. The committed files also
carry the manual review (`final`, `final_source`, `manual_reason`), which a
re-run drops and `noise_effect.py` needs. After re-running, compare with
`git diff audit/data` and restore the two files with `git checkout`.
